from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
# If AVATAR_FACE is missing, cli.py will prompt you once.
# Default under models/ so it can sit next to other local artifacts.
AVATAR_FACE: str = os.getenv("AVATAR_FACE", "models/avatar_examples/avatar.jpg")

# Where Piper voices live (downloaded by root setup.py)
PIPER_VOICES_DIR: str = os.getenv("PIPER_VOICES_DIR", "models/voices")

PIPER_VOICE_MALE: str = os.getenv(
    "PIPER_VOICE_MALE",
    os.path.join(PIPER_VOICES_DIR, "pt_PT-tugao-medium.onnx"),
)
PIPER_VOICE_FEMALE: str = os.getenv(
    "PIPER_VOICE_FEMALE",
    os.path.join(PIPER_VOICES_DIR, "dii_pt-PT.onnx"),
)

# Default voice if nothing else is specified
PIPER_VOICE_DEFAULT: str = os.getenv("PIPER_VOICE_DEFAULT", PIPER_VOICE_FEMALE)

# --- SadTalker defaults (used as CLI defaults) ---
SADTALKER_PREPROCESS_DEFAULT: str = os.getenv("SADTALKER_PREPROCESS_DEFAULT", "crop")  # crop|full|extfull
SADTALKER_SIZE_DEFAULT: str = os.getenv("SADTALKER_SIZE_DEFAULT", "256")  # 256|512
SADTALKER_BATCH_SIZE_DEFAULT: str = os.getenv("SADTALKER_BATCH_SIZE_DEFAULT", "1")
SADTALKER_ENHANCER_DEFAULT: str = os.getenv("SADTALKER_ENHANCER_DEFAULT", "")  # "" or "gfpgan"

# --- Output ---
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
