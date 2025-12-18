import os
import sys
import uuid
import subprocess
import time
from pathlib import Path

from config import (
    AVATAR_FACE,
    PIPER_VOICE_DEFAULT,
    PIPER_VOICE_FEMALE,
    PIPER_VOICE_MALE,
    RESULTS_DIR,
    SADTALKER_BATCH_SIZE_DEFAULT,
    SADTALKER_ENHANCER_DEFAULT,
    SADTALKER_PREPROCESS_DEFAULT,
    SADTALKER_SIZE_DEFAULT,
)

from src.piper_tts import speak as piper_speak


PROJECT_ROOT = Path(__file__).resolve().parent


def prompt_path(prompt: str) -> str:
    while True:
        value = input(f"{prompt}: ").strip().strip('"').strip("'")
        if value:
            p = Path(value).expanduser()
            if p.exists():
                return str(p)
            print(f"Path not found: {p}. Try again.")
        else:
            print("Please enter a non-empty path.")


def prompt_choice(prompt: str, choices: dict, default: str) -> str:
    print(prompt)
    for key, label in choices.items():
        print(f"  {key}) {label}")
    while True:
        value = input(f"Select option [{default}]: ").strip()
        if not value:
            return default
        if value in choices:
            return value
        print("Invalid choice, try again.")


def main() -> None:
    print("=== SadTalker CLI (Piper TTS -> SadTalker) ===")
    print(f"Working directory: {PROJECT_ROOT}")

    # 1) Text input (no Vosk)
    while True:
        text = input("Type your message and press Enter: ").strip()
        if text:
            break
        print("Please type a non-empty message.")

    # 2) Choose Piper voice (Enter = default from config)
    piper_default_choice = "2"
    try:
        default_name = Path(PIPER_VOICE_DEFAULT).expanduser().name
        if default_name == Path(PIPER_VOICE_MALE).expanduser().name:
            piper_default_choice = "1"
        elif default_name == Path(PIPER_VOICE_FEMALE).expanduser().name:
            piper_default_choice = "2"
    except Exception:
        piper_default_choice = "2"

    voice_choices = {
        "1": "Male (Tuga)" + (" [default]" if piper_default_choice == "1" else ""),
        "2": "Female (Dii)" + (" [default]" if piper_default_choice == "2" else ""),
    }
    voice_choice = prompt_choice("Choose Piper voice", voice_choices, default=piper_default_choice)
    voice_model = PIPER_VOICE_MALE if voice_choice == "1" else PIPER_VOICE_FEMALE

    # 3) Avatar image (auto-fill from config)
    avatar_path = AVATAR_FACE
    if not Path(avatar_path).expanduser().exists():
        print(f"Default avatar not found at: {avatar_path}")
        avatar_path = prompt_path("Enter avatar image path (png/jpg)")

    # 4) SadTalker settings (Enter = defaults from config)
    preprocess_default_choice = (
        "1"
        if SADTALKER_PREPROCESS_DEFAULT == "crop"
        else ("2" if SADTALKER_PREPROCESS_DEFAULT == "full" else "3")
    )
    preprocess_choices = {
        "1": "crop   (portrait / face crop)" + (" [default]" if preprocess_default_choice == "1" else ""),
        "2": "full   (full image)" + (" [default]" if preprocess_default_choice == "2" else ""),
        "3": "extfull (full image, extended crop)" + (" [default]" if preprocess_default_choice == "3" else ""),
    }
    preprocess_choice = prompt_choice(
        "Choose image preprocessing mode",
        preprocess_choices,
        default=preprocess_default_choice,
    )

    if preprocess_choice == "1":
        preprocess = "crop"
    elif preprocess_choice == "2":
        preprocess = "full"
    else:
        preprocess = "extfull"

    size_default_choice = SADTALKER_SIZE_DEFAULT if SADTALKER_SIZE_DEFAULT in {"256", "512"} else "256"
    size_choices = {
        "256": "256x256 (faster, less memory)" + (" [default]" if size_default_choice == "256" else ""),
        "512": "512x512 (slower, more memory)" + (" [default]" if size_default_choice == "512" else ""),
    }
    size_choice = prompt_choice("Choose output resolution", size_choices, default=size_default_choice)
    size = size_choice

    enhancer_default_choice = "2" if SADTALKER_ENHANCER_DEFAULT == "gfpgan" else "1"
    enhancer_choices = {
        "1": "No enhancer" + (" [default]" if enhancer_default_choice == "1" else ""),
        "2": "Yes, GFPGAN" + (" [default]" if enhancer_default_choice == "2" else ""),
    }
    enhancer_choice = prompt_choice("Use face enhancer (GFPGAN)?", enhancer_choices, default=enhancer_default_choice)
    enhancer = "gfpgan" if enhancer_choice == "2" else None

    batch_default = SADTALKER_BATCH_SIZE_DEFAULT if str(SADTALKER_BATCH_SIZE_DEFAULT).isdigit() else "1"
    while True:
        raw = input(f"Batch size (facerender) [{batch_default}]: ").strip()
        if not raw:
            batch_size = batch_default
            break
        if raw.isdigit() and int(raw) > 0:
            batch_size = raw
            break
        print("Invalid batch size. Enter a positive integer, or press Enter for default.")

    still = preprocess.startswith("full")

    result_root = PROJECT_ROOT / RESULTS_DIR
    result_root.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    result_dir = result_root / run_id

    # Use the same timestamp format as inference.py so the on-disk structure matches.
    timestamp = time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime())
    save_dir = result_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5) Generate wav via Piper inside the generated save_dir so it is
    # cleaned up with the other intermediate files.
    wav_path = str(save_dir / "piper.wav")
    if not Path(voice_model).exists():
        raise FileNotFoundError(
            f"Piper voice model not found: {voice_model}\n"
            "Run: python setup.py --models-only, or place the .onnx under models/voices/"
        )
    piper_speak(text, wav_path, voice_model)

    cmd = [
        sys.executable,
        "-m",
        "src.inference",
        "--driven_audio",
        str(wav_path),
        "--source_image",
        str(avatar_path),
        "--result_dir",
        str(result_dir),
        "--save_dir",
        str(save_dir),
        "--size",
        size,
        "--batch_size",
        batch_size,
        "--preprocess",
        preprocess,
    ]

    if still:
        cmd.append("--still")

    if enhancer:
        cmd.extend(["--enhancer", enhancer])

    print("\nRunning SadTalker:")
    print(" ", " ".join(cmd))
    print()

    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    mp4_files = sorted(result_dir.rglob("*.mp4"), key=os.path.getmtime)
    if mp4_files:
        print(f"\nDone. Generated video: {mp4_files[-1]}")
    else:
        print(f"\nDone, but no mp4 found under {result_dir}. Check the results/ folder.")


if __name__ == "__main__":
    main()
