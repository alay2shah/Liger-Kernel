from functools import wraps

from liger_kernel.ops.utils import is_hip
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward

_LFM2_HIP_MAX_LOGITS_CHUNK_BYTES = 128 * 1024 * 1024


@wraps(qwen2_lce_forward)
def lce_forward(self, *args, **kwargs):
    """LFM2 causal-LM forward with its measured ROCm logits-chunk policy."""
    if is_hip():
        kwargs.setdefault("max_logits_chunk_bytes", _LFM2_HIP_MAX_LOGITS_CHUNK_BYTES)
    return qwen2_lce_forward(self, *args, **kwargs)
