"""Section 3: PyTorch. One module file per COMP_GUIDE subsection (3.01-3.19).

The browser has no PyTorch build, so these run on torch-lite (generator/shims/torch).
"""

from . import (m01_tensors, m02_broadcasting, m03_autograd, m04_data, m05_training,  # noqa: F401
               m06_custom_functions, m07_memory, m08_attention, m09_transformer, m10_hooks, m11_kv_cache,
               m12_lora, m13_amp, m14_profiler, m15_triton, m16_ddp, m17_fsdp_checkpointing, m18_compile,
               m19_dpo)
from . import m01_catalogs, m02_m18_catalogs, m05_catalogs  # noqa: F401,E402
