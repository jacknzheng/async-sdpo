"""vLLM backend: the rollout engine (receive side) and the NCCL transport (send side).

Both halves of the vLLM integration live here, and this is the ONLY module in the package
that imports vLLM. `train/trainer.py` used to import
`vllm.distributed.weight_transfer.nccl_engine` directly, which meant the trainer could not
run against any other engine; `NCCLWeightTransport` below is where that code moved.

WEIGHT SYNC uses vLLM's native `vllm.distributed.weight_transfer` API. The engine is
constructed with `WeightTransferConfig(backend="nccl")`, and the update is a four-call
transaction:

    pause_generation(mode="keep") -> start_weight_update() -> update_weights(...)  [once per bucket] -> finish_weight_update() -> resume_generation()

vLLM owns the group lifecycle, the receive side posts its own matching broadcasts from the
names/dtypes/shapes metadata carried in the request, and bucketing is supported natively
via `packed=True`. There is no worker extension and no private `load_weights` call, so
none of this rides on unversioned internals.

Ordering still matters: the receivers must have their `update_weights` RPC in flight before
the trainer sends, or the sender blocks on a broadcast nobody is listening for. The
orchestrator owns that interleaving -- see `run.py:sync_weights`.

Two generation details are correctness-critical (invariants 1 and 2 in
`train/backends/backend.py`):

1. `logprobs_mode="processed_logprobs"`. The IS ratio's denominator must be the log-prob
   under the distribution the sampler ACTUALLY drew from -- after temperature and any
   logit processors. vLLM's default (`raw_logprobs`) is pre-processing, which would bias
   every ratio by exactly the sampling transform: silent and systematic, not a crash.

2. `logprobs=0` returns exactly the sampled token's own log-prob, which is all the
   sampled-token SDPO estimator needs. vLLM's sampler concatenates the sampled token's
   logprob first, so it is always present.

VERSION PIN: written against vLLM 0.26.x, which is where the native weight-transfer API
(`init_weight_transfer_engine` / `start_weight_update` / `update_weights` /
`finish_weight_update`) lives. It does NOT exist in 0.11/0.12 -- those releases need the
older `collective_rpc` + `worker_extension_cls` path. Pin exactly.

Every vLLM import is LAZY (inside the method that needs it), so this module -- and the
package as a whole -- stays importable on a machine without vLLM, e.g. a dev laptop
running the test suite. `tests/test_rollout.py::test_module_imports_without_vllm_installed`
pins that.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

from train.config import Config
from data.diagnostics import artifact_event
from data.dataset import Task, build_prompt
from train.models import RolloutResult
from train.backends.backend import InferenceEngine, WeightBucket, WeightTransport

logger = logging.getLogger(__name__)

# vLLM uses CUDA in subprocesses, which requires spawn rather than fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# torchrun / torchelastic leak these into every child. vLLM's TP workers inherit them
# and then try to join the *trainer* rendezvous (WORLD_SIZE matches TP size on a 4+4
# box) instead of opening their own TCPStore -- that is the Baseten 4+4 hang:
# "client socket has timed out after 600000ms while trying to connect to 127.0.0.1:<port>".
_TORCHRUN_ENV_EXACT = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_TORCHRUN_ENV_PREFIXES = ("TORCHELASTIC_", "PET_")


def _torchrun_env_keys(env: dict[str, str] | None = None) -> list[str]:
    """Names currently in `env` that torchrun/elastic set. Used by the isolator and tests."""
    source = os.environ if env is None else env
    keys = [name for name in _TORCHRUN_ENV_EXACT if name in source]
    keys.extend(k for k in source if k.startswith(_TORCHRUN_ENV_PREFIXES))
    # Preserve order but drop duplicates (exact names can also match a prefix).
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


@contextmanager
def isolated_from_torchrun() -> Iterator[None]:
    """Strip torch.distributed launch env for the duration of vLLM engine construction.

    Spawn copies `os.environ` at Process.start(), so this must wrap the call that
    actually forks EngineCore / TP workers (`AsyncLLM.from_engine_args`). The parent
    MUST restore the env before `init_process_group` -- the trainer ranks still need
    RANK / WORLD_SIZE / MASTER_PORT.

    Do not change CUDA_VISIBLE_DEVICES here. The parent may initialize CUDA later on
    a trainer GPU; remapping visibility for spawn would also remap the parent if CUDA
    is already live, and trainer rank 0 would lose cuda:4.
    """
    saved = {key: os.environ.pop(key) for key in _torchrun_env_keys()}
    # Force the new TP TCPStore onto loopback. Without this, some workstation images
    # advertise a non-loopback NIC and the workers never find each other.
    host_ip_was_set = "VLLM_HOST_IP" in os.environ
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    try:
        yield
    finally:
        if not host_ip_was_set:
            os.environ.pop("VLLM_HOST_IP", None)
        os.environ.update(saved)


@contextmanager
def pinned_visible_device(gpu: int) -> Iterator[None]:
    """Restrict CUDA_VISIBLE_DEVICES to one physical index for a spawn, then restore.

    Must wrap the call that forks hint-engine workers, and must run before the parent
    initializes CUDA. Restore the full 0-7 map before trainer / rollout CUDA init so
    those processes still see every device.
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


class HintEngine:
    """Frozen one-GPU vLLM for error-conditioned hints. No weight sync.

    Not an InferenceEngine: it never joins the NCCL transfer group and never sees
    trainer weights. Lives on cuda:{hint.gpu} (default 7) for the life of the run.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = config.generator.hint.model
        self.engine = None
        self.tokenizer = None
        self._request_counter = 0

    def start(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            raise RuntimeError(
                "hint vLLM cannot be spawned after torch.distributed.init_process_group: "
                "workers inherit the trainer rendezvous and hang on TCPStore. Start the "
                "hint engine before initializing the trainer process group."
            )

        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=self.model,
            dtype=self.config.model.dtype,
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            gpu_memory_utilization=self.config.generator.engine.gpu_memory_utilization,
            enforce_eager=False,
            enable_prefix_caching=True,
            max_model_len=min(8192, self.config.generator.engine.max_model_len),
            seed=self.config.trainer.seed,
        )
        stripped = _torchrun_env_keys()
        started = time.monotonic()
        artifact_event(
            "vllm",
            "hint_engine_starting",
            model=self.model,
            gpu=self.config.generator.hint.gpu,
            tensor_parallel_size=1,
            stripped_torchrun_env=stripped,
        )
        try:
            with pinned_visible_device(self.config.generator.hint.gpu):
                with isolated_from_torchrun():
                    logger.info(
                        "starting hint engine on cuda:%d isolated from torchrun (stripped %s)",
                        self.config.generator.hint.gpu,
                        stripped or "nothing",
                    )
                    self.engine = AsyncLLM.from_engine_args(engine_args)
        except Exception as exc:
            artifact_event(
                "vllm",
                "hint_engine_start_failed",
                model=self.model,
                gpu=self.config.generator.hint.gpu,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        try:
            self.tokenizer = self.engine.get_tokenizer()
        except Exception:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
        logger.info("hint engine ready: %s on cuda:%d", self.model, self.config.generator.hint.gpu)
        artifact_event(
            "vllm",
            "hint_engine_ready",
            model=self.model,
            gpu=self.config.generator.hint.gpu,
            elapsed_seconds=time.monotonic() - started,
        )

    def _apply_chat_template(self, messages: list[dict]) -> str:
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    async def complete(self, system: str, user: str) -> str:
        if self.engine is None:
            raise RuntimeError("hint engine not started; call start() first")

        from vllm import SamplingParams

        prompt = self._apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        self._request_counter += 1
        request_id = f"hint-{self._request_counter}-{uuid.uuid4().hex[:6]}"
        started = time.monotonic()
        artifact_event(
            "vllm",
            "hint_generation_started",
            request_id=request_id,
            model=self.model,
            gpu=self.config.generator.hint.gpu,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.generator.hint.max_tokens,
        )
        final_output = None
        try:
            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling,
                request_id=request_id,
            ):
                final_output = output
        except Exception as exc:
            artifact_event(
                "vllm",
                "hint_generation_failed",
                request_id=request_id,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        if final_output is None or not final_output.outputs:
            artifact_event(
                "vllm",
                "hint_generation_failed",
                request_id=request_id,
                elapsed_seconds=time.monotonic() - started,
                error_type="RuntimeError",
                error="hint generation returned no output",
            )
            raise RuntimeError(f"no output for hint request {request_id}")
        text = (final_output.outputs[0].text or "").strip()
        artifact_event(
            "vllm",
            "hint_generation_succeeded",
            request_id=request_id,
            elapsed_seconds=time.monotonic() - started,
            completion_tokens=len(final_output.outputs[0].token_ids),
            finish_reason=getattr(final_output.outputs[0], "finish_reason", None),
        )
        return text

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        self.engine.shutdown()
        self.engine = None


class VLLMRolloutEngine(InferenceEngine):
    """Async wrapper over vLLM's AsyncLLM for off-policy trajectory generation.

    Implements the `InferenceEngine` ABC in `train/backends/backend.py`.
    """

    def __init__(self, config: Config, model: str | None = None, seed: int = 0) -> None:
        self.config = config
        self.model = model or config.model.model
        self.seed = seed
        self.engine = None
        self.tokenizer = None
        self.policy_version = 0
        self._weight_group_ready = False
        self._request_counter = 0

    def start(self) -> None:
        """Build the AsyncLLM engine. vLLM is imported lazily so the rest of the package
        stays importable on machines without it (e.g. the dev laptop running the tests).

        Must run BEFORE the trainer `init_process_group`. vLLM's TP workers are spawned
        from this process; if a default process group already exists (or torchrun's
        RANK/WORLD_SIZE/MASTER_PORT/TORCHELASTIC_* still sit in the environment) they
        try to join that rendezvous instead of creating their own TCPStore, and engine
        init hangs for 600s on 127.0.0.1. Clearing three env vars is not enough -- the
        elastic agent-store flag is what keeps workers pinned to torchrun's port.
        """
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            raise RuntimeError(
                "vLLM TP workers cannot be spawned after torch.distributed.init_process_group: "
                "they inherit the trainer rendezvous and hang on TCPStore. Start the rollout "
                "engine before initializing the trainer process group."
            )

        from vllm import AsyncEngineArgs
        from vllm.config import WeightTransferConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=self.model,
            dtype=self.config.model.dtype,
            tensor_parallel_size=self.config.generator.engine.n_rollout_gpus,
            distributed_executor_backend=(
                "mp" if self.config.generator.engine.n_rollout_gpus > 1 else "uni"
            ),
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            gpu_memory_utilization=self.config.generator.engine.gpu_memory_utilization,
            enforce_eager=False,
            enable_prefix_caching=True,
            max_model_len=self.config.generator.engine.max_model_len,
            disable_custom_all_reduce=(
                self.config.generator.engine.disable_custom_all_reduce
            ),
            seed=self.seed,
            # includes sampling adjustments: top_k, temperature, etc.
            logprobs_mode="processed_logprobs",
        )
        # Synchronous: from_engine_args is not a coroutine, and the engine core process
        # starts in __init__. There is no start()/await to call. Isolation must wrap
        # THIS call: spawn snapshots os.environ at Process.start().
        stripped = _torchrun_env_keys()
        started = time.monotonic()
        artifact_event(
            "vllm",
            "engine_starting",
            model=self.model,
            tensor_parallel_size=self.config.generator.engine.n_rollout_gpus,
            max_model_len=self.config.generator.engine.max_model_len,
            gpu_memory_utilization=(
                self.config.generator.engine.gpu_memory_utilization
            ),
            stripped_torchrun_env=stripped,
        )
        with isolated_from_torchrun():
            logger.info(
                "starting rollout engine isolated from torchrun (stripped %s)",
                stripped or "nothing",
            )
            try:
                self.engine = AsyncLLM.from_engine_args(engine_args)
            except Exception as exc:
                artifact_event(
                    "vllm",
                    "engine_start_failed",
                    model=self.model,
                    elapsed_seconds=time.monotonic() - started,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
        try:
            self.tokenizer = self.engine.get_tokenizer()
        except Exception:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
        logger.info(
            "rollout engine ready: %s on %d GPU(s)",
            self.model,
            self.config.generator.engine.n_rollout_gpus,
        )
        artifact_event(
            "vllm",
            "engine_ready",
            model=self.model,
            tensor_parallel_size=self.config.generator.engine.n_rollout_gpus,
            elapsed_seconds=time.monotonic() - started,
        )

    def _sampling_params(self):
        """Translate our backend-agnostic SamplingParams into vLLM's.

        This is the conversion boundary: train/config.py's SamplingParams imports no vLLM,
        so config stays cheap to import and a second backend translates the same fields
        its own way.
        """
        from vllm import SamplingParams

        sp = self.config.generator.sampling_params
        return SamplingParams(
            temperature=sp.temperature,
            top_p=sp.top_p,
            max_tokens=sp.max_tokens,
            logprobs=0,  # sampled token's own logprob only -- see module docstring
        )

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
        return self.tokenizer

    def _apply_chat_template(self, messages: list[dict], tools: list[dict] | None) -> str:
        tokenizer = self._get_tokenizer()
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    def encode_text(self, text: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        ids = tokenizer.encode(text, add_special_tokens=False)
        return list(ids)

    def tokenize_chat(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        return self.encode_text(self._apply_chat_template(messages, tools))

    async def _complete(self, prompt, request_id: str):
        """Run one vLLM generate call; return (prompt_token_ids, completion)."""
        from vllm import TokensPrompt

        started = time.monotonic()
        artifact_event(
            "vllm",
            "generation_started",
            request_id=request_id,
            policy_version=self.policy_version,
            prompt_kind="token_ids" if isinstance(prompt, list) else "text",
            prompt_size=len(prompt),
        )
        request_input = (
            TokensPrompt(prompt_token_ids=prompt)
            if isinstance(prompt, list)
            else prompt
        )
        final_output = None
        try:
            async for output in self.engine.generate(
                prompt=request_input,
                sampling_params=self._sampling_params(),
                request_id=request_id,
            ):
                final_output = output
        except Exception as exc:
            logger.exception("vLLM generation failed for request %s", request_id)
            artifact_event(
                "vllm",
                "generation_failed",
                request_id=request_id,
                policy_version=self.policy_version,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        if final_output is None:
            artifact_event(
                "vllm",
                "generation_failed",
                request_id=request_id,
                policy_version=self.policy_version,
                elapsed_seconds=time.monotonic() - started,
                error_type="RuntimeError",
                error="generation stream returned no output",
            )
            raise RuntimeError(f"no output for request {request_id}")
        if not final_output.outputs:
            artifact_event(
                "vllm",
                "generation_failed",
                request_id=request_id,
                policy_version=self.policy_version,
                elapsed_seconds=time.monotonic() - started,
                error_type="RuntimeError",
                error="generation finished with no candidate output",
            )
            raise RuntimeError(f"request {request_id} finished with no output (aborted?)")
        completion = final_output.outputs[0]
        artifact_event(
            "vllm",
            "generation_succeeded",
            request_id=request_id,
            policy_version=self.policy_version,
            elapsed_seconds=time.monotonic() - started,
            prompt_tokens=len(final_output.prompt_token_ids),
            completion_tokens=len(completion.token_ids),
            finish_reason=getattr(completion, "finish_reason", None),
            stop_reason=getattr(completion, "stop_reason", None),
        )
        return list(final_output.prompt_token_ids), completion

    # ---- generation ----

    async def generate(self, task: Task, prompt_token_ids: list[int] | None = None) -> RolloutResult:
        if self.engine is None:
            raise RuntimeError("engine not started; call start() first")

        if prompt_token_ids is None:
            request_input = build_prompt(task, hint=None)
        else:
            request_input = prompt_token_ids

        self._request_counter += 1
        request_id = f"{task.task_id}-{self._request_counter}-{uuid.uuid4().hex[:6]}"
        prompt_ids, completion = await self._complete(request_input, request_id)
        return RolloutResult(
            task_id=task.task_id,
            prompt_token_ids=prompt_ids,
            response_token_ids=list(completion.token_ids),
            logprobs=self._extract_sampled_logprobs(completion),
            text=completion.text,
        )

    async def generate_tir(self, task: Task) -> RolloutResult:
        """Multi-turn TIR: train on vLLM-sampled tokens, mask env/search injections."""
        if self.engine is None:
            raise RuntimeError("engine not started; call start() first")
        from data.tau_harness import parse_tool_calls
        from data.tau_harness import AgentTurn

        async def generate_turn(messages: list[dict], tools: list[dict] | None) -> AgentTurn:
            prompt_str = self._apply_chat_template(messages, tools)
            self._request_counter += 1
            request_id = f"{task.task_id}-{self._request_counter}-{uuid.uuid4().hex[:6]}"
            prompt_ids, completion = await self._complete(prompt_str, request_id)
            text = completion.text or ""
            tool_calls = parse_tool_calls(text)
            content = None if tool_calls else text
            return AgentTurn(
                token_ids=list(completion.token_ids),
                logprobs=self._extract_sampled_logprobs(completion),
                content=content,
                tool_calls=tool_calls,
                prompt_token_ids=prompt_ids,
            )

        if task.domain:
            from data.tau_harness import default_user_llm_args, run_tau2_episode

            episode = await run_tau2_episode(
                task,
                generate_turn=generate_turn,
                encode=self.encode_text,
                tokenize_chat=self.tokenize_chat,
                user_llm=self.config.data.user_llm,
                user_llm_args=default_user_llm_args(),
                max_steps=self.config.generator.max_steps,
            )
            text = episode.transcript
            score = episode.reward
        else:
            from data.diligence_harness import make_search_executor, run_diligence_episode

            episode = await run_diligence_episode(
                task,
                generate_turn=generate_turn,
                encode=self.encode_text,
                tokenize_chat=self.tokenize_chat,
                execute_tool=make_search_executor(
                    mode=self.config.data.search_mode,
                    timeout=self.config.data.search_timeout,
                    max_chars=self.config.data.search_max_chars,
                    client_model=self.model,
                ),
                max_steps=self.config.generator.max_steps,
            )
            text = episode.transcript
            score = episode.reward if episode.reward else None

        artifact_event(
            "rollouts",
            "episode_completed",
            dataset="tau2" if task.domain else "diligence",
            task_id=task.task_id,
            domain=task.domain,
            policy_version=self.policy_version,
            termination=episode.termination,
            reward=episode.reward,
            prompt_tokens=len(episode.prompt_token_ids),
            response_tokens=len(episode.response_token_ids),
            sampled_tokens=sum(episode.loss_mask),
            injected_tokens=len(episode.loss_mask) - sum(episode.loss_mask),
            step_spans=episode.step_spans,
            transcript=episode.transcript,
            messages=episode.messages,
        )
        return RolloutResult(
            task_id=task.task_id,
            prompt_token_ids=episode.prompt_token_ids,
            response_token_ids=episode.response_token_ids,
            logprobs=episode.rollout_logprobs,
            text=text,
            loss_mask=episode.loss_mask,
            step_spans=episode.step_spans,
            judge_score=score,
        ) 
    # ---- native weight sync (receive side) ----

    async def init_weight_update_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int
    ) -> None:

        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferInitInfo

        await asyncio.wait_for(
            self.engine.init_weight_transfer_engine(
                WeightTransferInitRequest(
                    init_info=asdict(
                        NCCLWeightTransferInitInfo(
                            master_address=master_address,
                            master_port=master_port,
                            rank_offset=rank_offset,
                            world_size=world_size,
                        )
                    )
                )
            ),
            timeout=self.config.generator.engine.weight_sync_timeout_s,
        )
        self._weight_group_ready = True
        logger.info("joined weight-sync group at rank offset %d", rank_offset)

    async def pause_for_update(self) -> None:
        """Freeze generation and open the weight-update transaction.

        `mode="keep"` suspends in-flight rollouts rather than aborting them, so a long
        generation survives the swap and resumes under the NEW weights.
        """
        if not self._weight_group_ready:
            raise RuntimeError("weight sync not initialized; call init_weight_update_group")
        await self.engine.pause_generation(mode="keep")
        await self.engine.start_weight_update()

    async def receive_weight_bucket(
        self, names: list[str], dtype_names: list[str], shapes: list[list[int]]
    ) -> None:
        """Post the receive side for one bucket of weights.

        The receivers derive their matching broadcasts from this metadata, so the trainer's
        send order must match `names` exactly. Must be IN FLIGHT before the trainer sends
        -- the orchestrator handles that interleaving.
        """
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferUpdateInfo

        await self.engine.update_weights(
            WeightTransferUpdateRequest(
                update_info=asdict(
                    NCCLWeightTransferUpdateInfo(
                        names=names, dtype_names=dtype_names, shapes=shapes, packed=True,
                    )
                )
            )
        )

    async def finish_update(self, new_version: int) -> None:
        """Close the transaction and resume generation under the new weights."""
        await self.engine.finish_weight_update()
        # We'll reset the prefix cache so that our newly generated sequences are entirely from this policy generation, instead of splicing into 2 policies
        await self.engine.reset_prefix_cache()
        await self.engine.resume_generation()
        self.policy_version = new_version

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        self.engine.shutdown()  # synchronous, not a coroutine
        self.engine = None

    @staticmethod
    def _extract_sampled_logprobs(completion) -> list[float]:
        """Pull the log-prob of each SAMPLED token, aligned 1:1 with `token_ids`.

        vLLM returns `logprobs` as one dict per generated position, mapping token_id ->
        Logprob. With `logprobs=0` the sampled token is the only entry, but we look it up by id
        rather than taking the sole value, so this stays correct if someone raises `logprobs`
        to inspect alternatives -- with logprobs>0 the dict holds alternatives too, and taking
        the most likely one would bias every ratio toward 1. A missing entry would silently
        misalign the IS ratio, so it raises instead.
        """
        if completion.logprobs is None:
            raise ValueError(
                "rollout returned no logprobs; SamplingParams(logprobs=...) must be set -- "
                "they cannot be recomputed later without destroying the off-policy correction"
            )

        out: list[float] = []
        for position, token_id in enumerate(completion.token_ids):
            entry = completion.logprobs[position].get(token_id)
            if entry is None:
                raise ValueError(f"sampled token {token_id} missing from its logprob dict")
            out.append(entry.logprob)

        if len(out) != len(completion.token_ids):
            raise ValueError(
                f"extracted {len(out)} logprobs for {len(completion.token_ids)} tokens"
            )
        return out



class NCCLWeightTransport(WeightTransport):
    """Send side of weight sync, over vLLM's native NCCL weight-transfer group.

    Implements the `WeightTransport` ABC in `train/backends/backend.py`. This code used to
    live in `train/trainer.py`; it is here so the trainer holds no engine-specific imports
    and can be paired with any transport.
    """

    def __init__(self) -> None:
        self._group = None

    def setup(self, master_address: str, master_port: int, world_size: int) -> None:
        """Join the weight-transfer group as the SENDER at rank 0.

        `world_size` counts vLLM GPU workers plus one for the trainer -- each rollout engine
        contributes `tensor_parallel_size` ranks, not one rank per engine. This blocks until
        every rank has joined, so the rollout engines must already be initializing their
        side (the orchestrator issues both concurrently).
        """
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        logger.info(
            "weight-sync: opening rendezvous at %s:%d, waiting for %d receiver(s)",
            master_address, master_port, world_size - 1,
        )
        self._group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            )
        )
        logger.info("weight-sync: rendezvous complete, all %d ranks joined", world_size)

    def send_bucket(self, bucket: WeightBucket) -> None:
        """Broadcast one bucket over the weight-sync group.

        The receivers post their matching broadcasts from the metadata in their
        `update_weights` request, so send order must match `bucket.names` exactly. Their RPC
        must already be in flight when this is called, or this blocks on a broadcast nobody
        is listening for.
        """
        if self._group is None:
            raise RuntimeError("weight sync not initialized; call setup first")

        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=zip(bucket.names, bucket.tensors),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self._group, packed=True),
        )

    def teardown(self) -> None:
        """Drop the group handle. vLLM owns the group lifecycle, so there is nothing to
        close explicitly -- this only makes a torn-down transport fail loudly if reused."""
        self._group = None