from __future__ import annotations

import os
import subprocess
from typing import Optional


def speak(text: str, output_wav_path: str, voice_model: str, *, extra_args: Optional[list[str]] = None) -> str:
    """Generate speech audio using the `piper` CLI.

    Args:
        text: Text to synthesize.
        output_wav_path: Absolute or relative path to write the WAV file.
        voice_model: Path to the Piper .onnx model.
        extra_args: Optional extra args passed to piper.

    Returns:
        The output_wav_path.
    """

    out_dir = os.path.dirname(output_wav_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = ["piper", "-m", voice_model, "-t", text, "-f", output_wav_path]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Piper failed. "
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )

    if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
        raise RuntimeError(f"Empty audio file generated: {output_wav_path}")

    return output_wav_path
