"""
Diarized Transcript Tool
========================
Combines Apple Dictation text with pyannote.audio speaker diarization,
then uses a local LM Studio model to correct the transcript.

Usage:
    python diarize_transcript.py --audio session.m4a --transcript transcript.txt

    # Skip LLM correction step:
    python diarize_transcript.py --audio session.m4a --transcript transcript.txt --no-llm

    # Use a different LM Studio model or address:
    python diarize_transcript.py --audio session.m4a --transcript transcript.txt \
        --lm-url http://127.0.0.1:1234 --lm-model openai/gpt-oss-120b

Output:
    session_diarized.txt         — turn-by-turn transcript with [CLINICIAN] / [CHILD] labels
    session_diarized.json        — structured JSON for further analysis
    session_diarized_corrected.txt — LLM-corrected final transcript (if LM Studio is running)

Requirements:
    pip install pyannote.audio torch torchaudio pydub openai
    A free Hugging Face token (see setup instructions)
    LM Studio running locally with a model loaded (https://lmstudio.ai)
"""

import os
os.environ["PYANNOTE_TELEMETRY"] = "off"
import argparse
import json
import re
from pathlib import Path


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN APPLE DICTATION TRANSCRIPT
# ─────────────────────────────────────────────

def load_transcript(path: str) -> list[str]:
    """
    Load Apple Dictation text file.
    Splits into utterances on Cantonese/Chinese punctuation and newlines.

    Apple Dictation (macOS Ventura+) automatically adds punctuation, so raw
    paste output like:
        好，今日我哋玩下呢個。你叫咩名呀？我叫阿明。
    is split into individual utterances without any manual editing.

    Splits on:
        。 — full stop
        ！ — exclamation mark
        ？ — question mark
        … — ellipsis
        \n — manual line breaks (if any)

    Note: 、（comma-like pause）and ，（mid-sentence comma）are NOT used as
    split points because Apple Dictation uses them within a single utterance
    (e.g. "好，我明白" is one utterance). Splitting on commas would create
    too many fragments and hurt alignment accuracy.

    Returns a list of utterance strings.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on sentence-ending punctuation and newlines
    # Keeps the punctuation attached to the preceding utterance for readability
    parts = re.split(r"(?<=[。！？…])\s*|\n+", text)
    utterances = [u.strip() for u in parts if u.strip()]
    return utterances


# ─────────────────────────────────────────────
# 2. RUN PYANNOTE DIARIZATION ON AUDIO
# ─────────────────────────────────────────────

def diarize_audio(audio_path: str, hf_token: str) -> list[dict]:
    """
    Run pyannote.audio speaker diarization.
    Returns a list of segments:
        [{"start": 0.0, "end": 2.3, "speaker": "SPEAKER_00"}, ...]
    """
    from pyannote.audio import Pipeline

    print("Loading pyannote diarization model (first run downloads ~1GB)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("Using GPU.")
        else:
            print("Using CPU (may take a few minutes for long recordings).")
    except Exception:
        pass

    print(f"Diarizing {audio_path} ...")
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end":   round(turn.end, 2),
            "speaker": speaker
        })

    print(f"Found {len(set(s['speaker'] for s in segments))} speakers, "
          f"{len(segments)} segments.")
    return segments


# ─────────────────────────────────────────────
# 3. IDENTIFY CLINICIAN vs CHILD
# ─────────────────────────────────────────────

def identify_roles(segments: list[dict]) -> dict[str, str]:
    """
    Heuristic: speaker with MORE total speaking time = clinician.
    Speaker with LESS total time = child.
    Works well for typical SLP sessions.

    Returns: {"SPEAKER_00": "CLINICIAN", "SPEAKER_01": "CHILD"}
    """
    from collections import defaultdict

    speaker_duration = defaultdict(float)
    for seg in segments:
        speaker_duration[seg["speaker"]] += seg["end"] - seg["start"]

    ranked = sorted(speaker_duration.items(), key=lambda x: x[1], reverse=True)

    role_map = {}
    labels = ["CLINICIAN", "CHILD"] + [f"UNKNOWN_{i}" for i in range(1, 20)]
    for i, (speaker, duration) in enumerate(ranked):
        role = labels[i] if i < len(labels) else f"SPEAKER_{i}"
        role_map[speaker] = role
        print(f"  {speaker} -> {role}  ({duration:.1f}s total speaking time)")

    return role_map


# ─────────────────────────────────────────────
# 4. GET AUDIO DURATION
# ─────────────────────────────────────────────

def get_audio_duration(audio_path: str) -> float:
    """Return total duration of audio file in seconds."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception:
        return None


# ─────────────────────────────────────────────
# 5. ALIGN UTTERANCES TO SPEAKER SEGMENTS
# ─────────────────────────────────────────────

def align_utterances(
    utterances: list[str],
    segments: list[dict],
    role_map: dict[str, str],
    audio_duration: float = None
) -> list[dict]:
    """
    Apple Dictation provides no timestamps, so utterances are distributed
    evenly across the audio timeline. Each utterance is then assigned to
    the speaker segment with the most overlap.

    Accuracy improves when:
    - Utterances are short (one sentence each)
    - Speakers take clear turns (minimal overlapping speech)

    Returns list of dicts with utterance, role, speaker_id, approx timestamps.
    """
    if not utterances or not segments:
        return []

    if audio_duration is None:
        audio_duration = segments[-1]["end"]

    utt_duration = audio_duration / len(utterances)
    utt_windows = []
    for i, utt in enumerate(utterances):
        start = i * utt_duration
        end = start + utt_duration
        utt_windows.append((start, end, utt))

    results = []
    for utt_start, utt_end, utt_text in utt_windows:
        best_speaker = None
        best_overlap = 0.0

        for seg in segments:
            overlap_start = max(utt_start, seg["start"])
            overlap_end   = min(utt_end,   seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        role = role_map.get(best_speaker, "UNKNOWN") if best_speaker else "UNKNOWN"
        results.append({
            "utterance": utt_text,
            "role": role,
            "speaker_id": best_speaker,
            "approx_start": round(utt_start, 2),
            "approx_end":   round(utt_end, 2)
        })

    return results


# ─────────────────────────────────────────────
# 6. LM STUDIO CORRECTION
# ─────────────────────────────────────────────

def correct_with_lmstudio(
    diarized_text: str,
    dictation_text: str,
    lm_url: str = "http://127.0.0.1:1234",
    lm_model: str = "openai/gpt-oss-120b"
) -> str:
    """
    Use a local LM Studio model to correct the diarized Whisper transcript
    using Apple Dictation text as a reference hint.

    The LLM is asked to:
    - Fix proper nouns, names, and Cantonese-specific terms
    - Use Apple Dictation as a reference for words Whisper got wrong
    - Preserve all speaker labels ([CLINICIAN] / [CHILD]) and timestamps exactly
    - Not add or remove any turns

    Returns the corrected transcript as a string.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not found. Run: pip install openai")
        return None

    print(f"  Connecting to LM Studio at {lm_url} ...")
    client = OpenAI(base_url=f"{lm_url}/v1", api_key="lm-studio")

    system_prompt = (
        "You are a professional transcript editor specialising in Cantonese Chinese. "
        "You will be given:\n"
        "1. A diarized transcript produced by Whisper + pyannote (may contain errors in "
        "proper nouns, names, Cantonese romanisation, or domain-specific terms).\n"
        "2. An Apple Dictation reference text of the same recording (accurate wording "
        "but no speaker labels or timestamps).\n\n"
        "Your task:\n"
        "- Correct errors in the diarized transcript using the Apple Dictation text as a hint.\n"
        "- Preserve ALL speaker labels (e.g. [CLINICIAN], [CHILD]) exactly as-is.\n"
        "- Preserve ALL timestamp markers exactly as-is.\n"
        "- Do NOT add, remove, or reorder any turns.\n"
        "- Only fix wording — do not paraphrase or restructure sentences.\n"
        "- Output ONLY the corrected transcript, nothing else."
    )

    user_prompt = (
        f"=== DIARIZED TRANSCRIPT (Whisper + pyannote) ===\n{diarized_text}\n\n"
        f"=== APPLE DICTATION REFERENCE ===\n{dictation_text}\n\n"
        "Please output the corrected transcript now."
    )

    print(f"  Sending to model: {lm_model} ...")
    try:
        response = client.chat.completions.create(
            model=lm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,  # low temperature for factual correction
        )
        corrected = response.choices[0].message.content.strip()
        print("  LLM correction complete.")
        return corrected
    except Exception as e:
        print(f"  LM Studio request failed: {e}")
        print("  Make sure LM Studio is running and a model is loaded.")
        return None


# ─────────────────────────────────────────────
# 7. FORMAT & SAVE OUTPUT
# ─────────────────────────────────────────────

def format_transcript(aligned: list[dict]) -> str:
    """Format aligned transcript as readable text with role labels."""
    lines = []
    prev_role = None

    for item in aligned:
        role = item["role"]
        text = item["utterance"]

        if prev_role and role != prev_role:
            lines.append("")

        lines.append(f"[{role}]  {text}")
        prev_role = role

    return "\n".join(lines)


def save_outputs(aligned: list[dict], audio_path: str, corrected_text: str = None):
    """Save .txt, .json, and optionally corrected .txt outputs next to the audio file."""
    base = Path(audio_path).stem
    out_dir = Path(audio_path).parent

    txt_path = out_dir / f"{base}_diarized.txt"
    txt_path.write_text(format_transcript(aligned), encoding="utf-8")
    print(f"\nSaved transcript:           {txt_path}")

    json_path = out_dir / f"{base}_diarized.json"
    json_path.write_text(
        json.dumps(aligned, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved JSON:                 {json_path}")

    corrected_path = None
    if corrected_text:
        corrected_path = out_dir / f"{base}_diarized_corrected.txt"
        corrected_path.write_text(corrected_text, encoding="utf-8")
        print(f"Saved corrected transcript: {corrected_path}")

    return txt_path, json_path, corrected_path


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine Apple Dictation transcript with pyannote speaker diarization, "
                    "then optionally correct with a local LM Studio model."
    )
    parser.add_argument("--audio",      required=True,  help="Path to audio file (.m4a, .mp3, .wav, .webm)")
    parser.add_argument("--transcript", required=True,  help="Path to Apple Dictation .txt file")
    parser.add_argument("--token",      default=None,   help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--flip-roles", action="store_true",
                        help="Swap CLINICIAN/CHILD if auto-detection got them backwards")
    parser.add_argument("--no-llm",     action="store_true",
                        help="Skip the LM Studio correction step")
    parser.add_argument("--lm-url",     default="http://127.0.0.1:1234",
                        help="LM Studio server URL (default: http://127.0.0.1:1234)")
    parser.add_argument("--lm-model",   default="openai/gpt-oss-120b",
                        help="LM Studio model identifier (default: openai/gpt-oss-120b)")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token required. Pass --token YOUR_TOKEN "
            "or set the HF_TOKEN environment variable.\n"
            "Get a free token at: https://huggingface.co/settings/tokens\n"
            "Then accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    print(f"\n[1/6] Loading transcript from {args.transcript}")
    utterances = load_transcript(args.transcript)
    dictation_raw = Path(args.transcript).read_text(encoding="utf-8")
    print(f"      Found {len(utterances)} utterances.")

    print(f"\n[2/6] Running speaker diarization on {args.audio}")
    segments = diarize_audio(args.audio, hf_token)

    print("\n[3/6] Identifying speaker roles:")
    role_map = identify_roles(segments)

    if args.flip_roles:
        speakers = list(role_map.keys())
        if len(speakers) >= 2:
            role_map[speakers[0]], role_map[speakers[1]] = \
                role_map[speakers[1]], role_map[speakers[0]]
            print("      Roles flipped as requested.")

    print("\n[4/6] Measuring audio duration...")
    duration = get_audio_duration(args.audio)
    if duration:
        print(f"      Duration: {duration:.1f}s")

    print("\n[5/6] Aligning utterances to speakers...")
    aligned = align_utterances(utterances, segments, role_map, duration)
    diarized_text = format_transcript(aligned)

    # ── LM Studio correction ──────────────────────────────────────────────
    corrected_text = None
    if not args.no_llm:
        print(f"\n[6/6] Correcting transcript with LM Studio ({args.lm_model})...")
        corrected_text = correct_with_lmstudio(
            diarized_text=diarized_text,
            dictation_text=dictation_raw,
            lm_url=args.lm_url,
            lm_model=args.lm_model
        )
        if corrected_text is None:
            print("      LLM correction skipped — proceeding with uncorrected transcript.")
    else:
        print("\n[6/6] LLM correction skipped (--no-llm flag set).")

    save_outputs(aligned, args.audio, corrected_text)

    print("\n--- TRANSCRIPT PREVIEW (first 10 utterances) ---\n")
    preview = corrected_text if corrected_text else diarized_text
    for line in preview.splitlines()[:10]:
        print(line)
    remaining = len(preview.splitlines()) - 10
    if remaining > 0:
        print(f"... ({remaining} more lines)")

    print("\nDone.")


if __name__ == "__main__":
    main()
