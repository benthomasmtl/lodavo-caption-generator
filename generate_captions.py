#!/usr/bin/env python3
"""
generate_captions.py
Lodavo's SRT caption generator for CapCut with perfect timing and positioning.

Focus: Maximum accuracy, short phrases, sentence breaks, all caps, question marks only.
"""

import argparse
import re
from pathlib import Path


def secs_to_srt_time(secs: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    seconds = int(secs % 60)
    millis = int((secs % 1) * 1000)
    return f"{hours:02d}:{mins:02d}:{seconds:02d},{millis:03d}"


EMOJI_MAP = {
    "save": "💰", "savings": "💰", "money": "💸", "cash": "💸",
    "win": "🏆", "winner": "🏆", "prize": "🎉",
    "bank": "🏦", "goal": "🎯", "free": "🆓",
    "today": "📅", "week": "📆", "weekly": "📆",
    "ticket": "🎟️", "tickets": "🎟️", "bonus": "➕",
    "jackpot": "💥", "lodavo": "💙", "lodevo": "💙"
}

EMPHASIS_WORDS = {
    "save", "savings", "money", "cash", "win", "winner", "prize",
    "free", "jackpot", "tickets", "bonus", "lodavo", "lodevo"
}


def should_add_emoji_for_word(word: str, emoji_counter: dict) -> bool:
    """Determine if we should add emoji based on alternating logic."""
    if word not in emoji_counter:
        emoji_counter[word] = 0
    
    emoji_counter[word] += 1
    
    # Add emoji on 1st, 3rd, 5th occurrence, etc.
    return emoji_counter[word] % 2 == 1


def add_emojis_inline(text: str, emoji_counter: dict, recent_emoji_count: int) -> tuple:
    """Add emojis right after emphasis words, with frequency control."""
    # Don't add if we just had an emoji recently
    if recent_emoji_count > 0:
        return text, recent_emoji_count - 1
    
    words = text.split()
    result_words = []
    emoji_added = False
    
    for word in words:
        result_words.append(word)
        
        # Check if this word (without punctuation) matches emoji keywords
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in EMOJI_MAP and not emoji_added:
            if should_add_emoji_for_word(clean_word, emoji_counter):
                result_words.append(EMOJI_MAP[clean_word])
                emoji_added = True
    
    result_text = ' '.join(result_words)
    new_recent_count = 2 if emoji_added else 0  # Skip next 2 captions if we added emoji
    
    return result_text, new_recent_count


def fix_spelling(text: str) -> str:
    """Fix common spelling issues for Lodavo brand."""
    # Fix Lodavo spelling variations
    text = re.sub(r'\blodevo\b', 'lodavo', text, flags=re.IGNORECASE)
    text = re.sub(r'\blodevo\'s\b', 'lodavo\'s', text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Clean and format text: all caps, question marks only, no other punctuation."""
    # Fix spelling first
    text = fix_spelling(text)
    
    # Convert to uppercase
    text = text.upper()
    
    # Basic comma heuristics BEFORE stripping punctuation:
    # Add commas after common introductory words/phrases if followed by a space and a word.
    intro_words = [
        'look', 'listen', 'hey', 'wait', 'so', 'now', 'okay', 'ok', 'well', 'basically'
    ]
    def add_intro_commas(match):
        word = match.group(1)
        rest = match.group(2)
        return f"{word}, {rest}" if rest else f"{word},"
    pattern = re.compile(r'\b(' + '|'.join(intro_words) + r')\b\s+([^,])', flags=re.IGNORECASE)
    # Apply iteratively a couple times to catch layered matches
    for _ in range(2):
        text = pattern.sub(lambda m: f"{m.group(1)}, {m.group(2)}", text)

    # Keep only question marks, remove other punctuation except spaces and apostrophes and commas we just inserted
    text = re.sub(r'[^\w\s\',?]', '', text)
    
    # Ensure single spaces after commas
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+,', ',', text)  # remove leading spaces before commas
    text = re.sub(r',\s*', ', ', text)
    
    # Remove commas immediately before emojis (no comma before emoji)
    # Match comma + space + emoji pattern
    text = re.sub(r',\s+([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF]+)', r' \1', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def split_by_sentences(text: str) -> list:
    """Split text by sentences, keeping question marks."""
    # Split on sentence endings but keep the question mark
    sentences = re.split(r'(\?)', text)
    
    result = []
    current = ""
    
    for i, part in enumerate(sentences):
        if part == '?':
            if current.strip():
                result.append(current.strip() + '?')
                current = ""
        else:
            current += part
    
    # Add any remaining text as a sentence
    if current.strip():
        result.append(current.strip())
    
    return [s for s in result if s.strip()]


def break_into_short_phrases(text: str, max_words: int = 4) -> list:
    """Break text into short phrases of max_words or fewer.

    If an emphasis word (that will likely receive an emoji) appears, it becomes its own
    single-word phrase so we can render it standalone with its emoji for visual punch.
    """
    words = text.split()
    phrases = []
    current = []

    def flush_current():
        if current:
            phrases.append(' '.join(current))
            current.clear()

    for w in words:
        base = re.sub(r'[^\w]', '', w.lower())
        # If emphasis word, flush any accumulated words and add alone.
        if base in EMPHASIS_WORDS:
            flush_current()
            phrases.append(w)
            continue

        current.append(w)
        if len(current) >= max_words:
            flush_current()

    flush_current()
    return phrases


def process_segments_minimal(segments, max_words_per_caption=4):
    """Process segments with sentence-based breaks and short phrases."""
    captions = []
    
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
            
        # Clean the text
        clean = clean_text(text)
        if not clean:
            continue
            
        # Split by sentences first
        sentences = split_by_sentences(clean)
        
        # Calculate timing per sentence based on segment duration
        seg_duration = seg["end"] - seg["start"]
        
        if len(sentences) == 1:
            # Single sentence - break into short phrases
            phrases = break_into_short_phrases(sentences[0], max_words_per_caption)
            phrase_duration = seg_duration / len(phrases)
            
            for i, phrase in enumerate(phrases):
                start_time = seg["start"] + (i * phrase_duration)
                end_time = seg["start"] + ((i + 1) * phrase_duration)
                captions.append({
                    "start": start_time,
                    "end": end_time,
                    "text": phrase
                })
        else:
            # Multiple sentences - each gets its own timing
            sentence_duration = seg_duration / len(sentences)
            
            for i, sentence in enumerate(sentences):
                phrases = break_into_short_phrases(sentence, max_words_per_caption)
                
                # Time allocation for this sentence
                sentence_start = seg["start"] + (i * sentence_duration)
                sentence_end = seg["start"] + ((i + 1) * sentence_duration)
                
                if len(phrases) == 1:
                    captions.append({
                        "start": sentence_start,
                        "end": sentence_end,
                        "text": phrases[0]
                    })
                else:
                    phrase_duration = sentence_duration / len(phrases)
                    for j, phrase in enumerate(phrases):
                        start_time = sentence_start + (j * phrase_duration)
                        end_time = sentence_start + ((j + 1) * phrase_duration)
                        captions.append({
                            "start": start_time,
                            "end": end_time,
                            "text": phrase
                        })
    
    return captions


def main():
    parser = argparse.ArgumentParser(description="Generate minimal, accurate SRT captions")
    parser.add_argument("input_audio", type=str, help="Input audio file")
    parser.add_argument("output_srt", type=str, help="Output SRT file")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model")
    parser.add_argument("--language", type=str, default="en", help="Language")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--vad", action="store_true", help="Enable VAD")
    parser.add_argument("--max-words", type=int, default=4, help="Max words per caption")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay captions by X seconds to prevent spoiling")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Please install Faster-Whisper first: pip install -r requirements_local.txt")
        return

    # Audio loading logic - use numpy/soundfile fallback since PyAV has build issues on Python 3.13
    use_numpy_audio = True
    try:
        import soundfile as sf
        import numpy as np
    except Exception:
        print("Missing audio packages. Install soundfile and numpy")
        return

    print(f"Loading Whisper model '{args.model}'...")
    compute_type = "auto"
    device = None if args.device == "auto" else args.device
    model = WhisperModel(args.model, device=device or "auto", compute_type=compute_type)

    print(f"Transcribing '{args.input_audio}' with high accuracy settings...")
    
    # Use higher beam size for better accuracy
    if not use_numpy_audio:
        segments_iter, info = model.transcribe(
            args.input_audio,
            language=args.language,
            beam_size=10,  # Higher for accuracy
            vad_filter=args.vad,
            word_timestamps=True,
            temperature=0.0,  # Deterministic for consistency
        )
    else:
        # Audio conversion logic
        audio_path = args.input_audio
        import tempfile
        import os
        import subprocess
        
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in ['.m4a', '.mp4', '.mov', '.aac', '.flac', '.ogg', '.opus', '.webm']:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav.close()
            subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-y',
                temp_wav.name
            ], check=True, capture_output=True)
            audio_path = temp_wav.name
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        data, sr = sf.read(audio_path, dtype='float32')
        
        if cleanup_temp:
            try:
                os.unlink(audio_path)
            except:
                pass
        
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        segments_iter, info = model.transcribe(
            data,
            language=args.language,
            beam_size=10,
            vad_filter=args.vad,
            word_timestamps=True,
            temperature=0.0,
        )

    # Collect segments
    print("Processing segments for maximum accuracy...")
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    print(f"Collected {len(segments)} segments. Creating short-phrase captions...")
    
    # Process into minimal captions
    captions = process_segments_minimal(segments, max_words_per_caption=args.max_words)

    # Generate SRT with timing delay and smart emoji placement
    srt_lines = []
    emoji_counter = {}
    recent_emoji_count = 0
    
    for i, cap in enumerate(captions, 1):
        start_time = secs_to_srt_time(cap["start"] + args.delay)
        end_time = secs_to_srt_time(cap["end"] + args.delay)
        
        # Add emojis with smart positioning and frequency control
        text_with_emojis, recent_emoji_count = add_emojis_inline(
            cap["text"], emoji_counter, recent_emoji_count
        )

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text_with_emojis)
        srt_lines.append("")  # Blank line between entries    # Write SRT file
    Path(args.output_srt).write_text("\n".join(srt_lines), encoding="utf-8")
    
    print(f"\n✅ Success! Generated {len(captions)} minimal captions in {args.output_srt}")
    print("\n📝 Import into CapCut:")
    print("  1. Import your video")
    print("  2. Go to Text → Captions → Import subtitle")
    print(f"  3. Select {args.output_srt}")
    print("  4. Manually adjust colors, stroke, and positioning as needed")
    print(f"\n💡 Caption style: ALL CAPS, max {args.max_words} words, question marks only")


if __name__ == "__main__":
    main()