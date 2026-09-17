"""Validate the ARM kernels and four-rank communication used by the full RB arm."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--collective", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != args.job_id:
        raise SystemExit("GPU validation must run inside the selected holder")
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expected four visible GPUs")
    if args.collective:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl")
        value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(value)
        assert value.item() == 10
        dist.barrier()
        dist.destroy_process_group()
        print(f"NCCL rank {rank} passed", flush=True)
        return
    from flash_attn import flash_attn_func

    devices = []
    for index in range(4):
        torch.cuda.set_device(index)
        q = torch.randn(2, 32, 4, 64, device=f"cuda:{index}", dtype=torch.float16, requires_grad=True)
        output = flash_attn_func(q, q, q, causal=True)
        output.float().square().mean().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
        compiled = torch.compile(lambda x: (x.sin() + x.square()).sum())
        x = torch.randn(128, device=f"cuda:{index}", requires_grad=True)
        compiled(x).backward()
        assert torch.isfinite(x.grad).all()
        torch.cuda.synchronize()
        devices.append(torch.cuda.get_device_name(index))
        del q, output, x, compiled
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "devices": devices,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flash_attention_backward": True,
                "triton_backward": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
