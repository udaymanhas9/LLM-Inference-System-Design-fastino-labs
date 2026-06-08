import asyncio
import random
from abc import ABC, abstractmethod
from typing import AsyncGenerator

_MOCK_VOCAB = [
    "The", "inference", "system", "processes", "tokens", "efficiently", "using",
    "PagedAttention", "and", "continuous", "batching", "to", "maximize", "GPU",
    "utilization", "while", "maintaining", "low", "latency", "for", "all", "tenants",
    ".", "The", "chunked", "prefill", "scheduler", "interleaves", "decode", "steps",
    "with", "prefill", "chunks", "to", "prevent", "starvation", "of", "active", "sequences",
]


class BaseEngine(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        ...


class MockEngine(BaseEngine):
    """
    Deterministic mock for local demo on CPU / M4.
    Simulates realistic TTFT and inter-token delay without requiring a GPU.

    Production: replace with VLLMEngine.
    """

    def __init__(
        self,
        tokens_per_second: float = 50.0,
        ttft_ms: float = 200.0,
        seed: int = 42,
    ) -> None:
        self._tokens_per_second = tokens_per_second
        self._ttft_ms = ttft_ms
        self._rng = random.Random(seed)

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        await asyncio.sleep(self._ttft_ms / 1000.0)

        n_tokens = min(max_tokens, self._rng.randint(30, 80))
        delay = 1.0 / self._tokens_per_second

        for i in range(n_tokens):
            word = self._rng.choice(_MOCK_VOCAB)
            yield (" " + word) if i > 0 else word
            await asyncio.sleep(delay)


class VLLMEngine(BaseEngine):
    """
    Production engine wrapping vllm.AsyncLLMEngine.

    Requires: pip install vllm  (CUDA environment, Linux/WSL2 only)
    Not usable on macOS — use MockEngine locally.
    """

    def __init__(self, model: str, tensor_parallel_size: int = 1) -> None:
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            args = AsyncEngineArgs(model=model, tensor_parallel_size=tensor_parallel_size)
            self._engine = AsyncLLMEngine.from_engine_args(args)

            from vllm import SamplingParams
            self._SamplingParams = SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm not installed or not available on this platform. "
                "Use MockEngine for local development."
            ) from exc

    async def generate(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        import uuid
        params = self._SamplingParams(max_tokens=max_tokens)
        request_id = str(uuid.uuid4())

        prev_len = 0
        async for output in self._engine.generate(prompt, params, request_id=request_id):
            if output.outputs:
                text = output.outputs[0].text
                delta = text[prev_len:]
                if delta:
                    yield delta
                prev_len = len(text)


def create_engine(use_mock: bool = True, **kwargs) -> BaseEngine:
    if use_mock:
        return MockEngine(**kwargs)
    model = kwargs.pop("model", "meta-llama/Llama-2-13b-chat-hf")
    return VLLMEngine(model=model, **kwargs)
