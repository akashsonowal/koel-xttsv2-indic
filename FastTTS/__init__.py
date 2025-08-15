import os
import sys
import logging
from pathlib import Path
import importlib.metadata as im

from FastTTS.utils.generic_utils import is_pytorch_at_least_2_4

log = logging.getLogger("boot")
if not log.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

def _read_version() -> str:
    """
    Try VERSION file next to this script, else APP_VERSION env var,
    else a sensible default.
    """
    try:
        v = (Path(__file__).with_name("VERSION")).read_text(encoding="utf-8").strip()
        if v:
            return v
    except Exception:
        pass
    return os.getenv("APP_VERSION", "0.0.0-dev")

# Always use your file/env for version (no package lookup)
__version__ = _read_version()

# Conflict check identical to code piece 2
if "coqpit" in im.packages_distributions().get("coqpit", []):
    msg = (
        "coqui-tts switched to a forked version of Coqpit, but you still have the original "
        "package installed. Run the following to avoid conflicts:\n"
        "  pip uninstall coqpit\n"
        "  pip install coqpit-config"
    )
    raise ImportError(msg)

# PyTorch 2.4+ registrations (same as code piece 2)
if is_pytorch_at_least_2_4():
    import _codecs  # noqa: F401
    from collections import defaultdict

    import numpy as np  # noqa: F401
    import torch
    from packaging import version  # noqa: F401

    from FastTTS.config.shared_configs import BaseDatasetConfig
    from FastTTS.tts.configs.xtts_config import XttsConfig
    from FastTTS.tts.models.xtts import XttsArgs, XttsAudioConfig
    from FastTTS.utils.radam import RAdam

    torch.serialization.add_safe_globals([dict, defaultdict, RAdam])

    # XTTS
    torch.serialization.add_safe_globals([BaseDatasetConfig, XttsConfig, XttsAudioConfig, XttsArgs])