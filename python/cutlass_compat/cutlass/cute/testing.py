"""cutlass.cute.testing — JitArguments + benchmark (compat).

benchmark() mirrors the official harness: rotate through `workspace_count`
fresh workspaces to defeat L2 caching, warm up, then time `iterations`
invocations with CUDA events; returns microseconds per invocation.
"""
from __future__ import annotations

import torch


class JitArguments:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.args


def benchmark(func, workspace_generator=None, workspace_count=1,
              stream=None, warmup_iterations=2, iterations=100,
              **kw) -> float:
    workspaces = [workspace_generator() for _ in range(workspace_count)] \
        if workspace_generator else [None]

    # warmup
    for i in range(warmup_iterations):
        ws = workspaces[i % len(workspaces)]
        func(*(ws.args if ws else ()))

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iterations):
        ws = workspaces[i % len(workspaces)]
        func(*(ws.args if ws else ()))
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations  # µs per call
