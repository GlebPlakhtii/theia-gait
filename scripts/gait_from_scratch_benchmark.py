"""Model-only timing for an already loaded assessment model.

No checkpoint/data loading or pose extraction occurs here. Call on a frozen
trained model for the final report; synthetic checks do not measure that result.
"""
from datetime import datetime, timezone
import time

import numpy as np
import torch


def benchmark_assessment(model, pose, context, *, warmup=10, repetitions=100):
    """Time synchronous, full-observed-window inference at the supplied batch size.

    Inputs and the model must already share a device. Timings include forward
    input validation and output construction, but exclude transfer, decoding,
    preprocessing, pose extraction, plotting and checkpoint loading. CUDA peak
    allocation is local to this process, not total device usage.
    """
    if type(warmup) is not int or type(repetitions) is not int or warmup < 1 or repetitions < 2:
        raise ValueError('Require at least one warmup and two timed repetitions')
    parameters = list(model.parameters())
    if not parameters or not isinstance(pose, torch.Tensor) or not isinstance(context, torch.Tensor):
        raise ValueError('Require a parameterized model and tensor inputs')
    device = parameters[0].device
    if device.type not in ('cpu', 'cuda') or pose.device != device or context.device != device:
        raise ValueError('Model and input tensors must share a supported CPU or CUDA device')
    if any(p.device != device for p in parameters) or any(b.device != device for b in model.buffers()):
        raise ValueError('A distributed model is not supported by this timing protocol')
    if pose.ndim != 4 or pose.shape[0] < 1 or context.ndim != 2 or len(context) != len(pose):
        raise ValueError('Expected a nonempty pose batch and aligned context')
    if not pose.is_floating_point() or pose.dtype != context.dtype or any(p.dtype != pose.dtype for p in parameters):
        raise ValueError('Model and inputs must share their floating-point precision')

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def check_outputs(outputs):
        if not isinstance(outputs, dict) or not outputs or any(
            not isinstance(x, torch.Tensor) or not torch.isfinite(x).all()
            for x in outputs.values()
        ):
            raise FloatingPointError('Benchmark outputs must be finite tensors')

    previous_modes = [(module, module.training) for module in model.modules()]
    durations = []
    baseline = peak = peak_reserved = None
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                output = model(pose, context)
                check_outputs(output)
                del output
            synchronize()
            if device.type == 'cuda':
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repetitions):
                synchronize()
                started = time.perf_counter()
                output = model(pose, context)
                synchronize()
                durations.append(time.perf_counter() - started)
                # Do not accumulate output tensors across repetitions.
                if len(durations) == repetitions:
                    if device.type == 'cuda':
                        peak = torch.cuda.max_memory_allocated(device)
                        peak_reserved = torch.cuda.max_memory_reserved(device)
                    check_outputs(output)
                del output
    finally:
        # Preserve even callers' intentionally mixed train/eval submodule modes.
        for module, previous in previous_modes:
            module.training = previous
    if not np.isfinite(durations).all() or min(durations) <= 0:
        raise RuntimeError('The timing clock returned an invalid duration')
    average = float(np.mean(durations))
    return dict(
        measured_at=datetime.now(timezone.utc).isoformat(),
        scope='Model forward only; full observed window, no future forecasting',
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
        precision=str(pose.dtype), batch_size=len(pose), input_shape=list(pose.shape),
        parameter_count=sum(p.numel() for p in parameters),
        trainable_parameter_count=sum(p.numel() for p in parameters if p.requires_grad),
        warmup_iterations=warmup, timed_iterations=repetitions,
        latency_ms=dict(mean=average * 1000, median=float(np.median(durations)) * 1000,
                        p95=float(np.quantile(durations, .95)) * 1000,
                        minimum=float(min(durations)) * 1000),
        windows_per_second=len(pose) / average,
        timed_durations_s=durations,
        cuda_memory=dict(baseline_allocated_bytes=baseline, peak_allocated_bytes=peak,
                         incremental_peak_allocated_bytes=peak - baseline if peak is not None else None,
                         peak_reserved_bytes=peak_reserved,
                         scope='Current process, including resident model and inputs; not whole-device usage'),
        excluded=['video decoding', 'pose extraction', 'window adapter', 'device transfer',
                  'checkpoint loading', 'plotting', 'training'],
        checkpoint_provenance='Caller must bind the trained checkpoint and input hashes in the final report',
    )
