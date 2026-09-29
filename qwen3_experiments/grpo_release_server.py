"""Record actual inference features and counters at rollout sleep boundaries."""

import json
import os
import time
from pathlib import Path

import aiohttp

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def inspect_engine(engine):
    config = engine.vllm_config
    return {
        "use_v2_model_runner": config.use_v2_model_runner,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "attention_backend": str(config.attention_config.backend),
        "flash_attn_version": config.attention_config.flash_attn_version,
        "enable_prefix_caching": config.cache_config.enable_prefix_caching,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
    }


class BenchmarkvLLMHttpServer(vLLMHttpServer):
    async def sleep(self):
        root = os.environ.get("GRPO_BENCHMARK_METRICS_DIR")
        if root:
            record = {"time": time.time(), "pid": os.getpid(), "replica_rank": self.replica_rank}
            try:
                # Read the resolved engine configuration locally. Sending a
                # Python function through collective_rpc requires insecure
                # pickle serialization in vLLM 0.24 and is unnecessary here.
                record["engine_config"] = inspect_engine(self.engine)
            except Exception as exc:
                record["config_probe_error"] = repr(exc)
            try:
                address, port = self.get_server_address()
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(f"http://{address}:{port}/metrics") as response:
                        record["metrics_http_status"] = response.status
                        record["prometheus"] = await response.text()
            except Exception as exc:
                record["metrics_probe_error"] = repr(exc)
            folder = Path(root)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"replica_{self.replica_rank}_{time.time_ns()}.json").write_text(json.dumps(record, indent=2))
        await super().sleep()
