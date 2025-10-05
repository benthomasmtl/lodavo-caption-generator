# Lodavo Caption Generator 💙

AI-powered caption generation tool built specifically for **Lodavo's video content creation workflow**. Creates engaging, accurate captions for short-form video content with perfect timing and CapCut optimization.

Built with OpenAI's Whisper for transcription and fine-tuned for Lodavo's branding and style requirements.

## ✨ Features

- **🎯 High Accuracy**: Uses Whisper large-v3 model with optimized settings
- **⚡ Smart Timing**: 0.3s delay by default to prevent caption spoiling
- **🎨 Lodavo Branding**: Automatic spelling correction and brand consistency
- **😀 Smart Emojis**: Context-aware emoji placement with frequency control
- **📱 CapCut Ready**: SRT format optimized for direct CapCut import
- **🧵 Standalone Emphasis**: Key words + emoji appear as their own single-word captions
- **🔧 Robust Setup**: Handles PyAV compatibility issues automatically

## 🚀 Quick Start

### Prerequisites

1. **Python 3.13** (installed via Homebrew on macOS)
2. **FFmpeg** (for audio conversion)
   ```bash
   brew install ffmpeg
   ```

### Installation

1. **Set up the virtual environment** (already configured):

   ```bash
   cd /path/to/lodavo-captions
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_local.txt
   ```

## 📝 Usage

### Basic Command

**⚠️ Important: Always activate the virtual environment first!**

```bash
source .venv/bin/activate
python generate_captions.py input_audio.m4a output.srt --model large-v3 --language en --vad --delay 0.3
```

### Parameters

- `input_audio.m4a` - Your audio/video file (.m4a, .mp3, .wav, .mov, etc.)
- `output.srt` - Output caption file name (SubRip)
- `--model large-v3` - Whisper model size (tiny/base/small/medium/large-v3)
  - **large-v3**: Best accuracy (~3GB download, recommended)
  - **small**: Fast, good accuracy (~500MB download)
- `--language en` - Language code (en for English)
- `--vad` - Enable Voice Activity Detection for cleaner segments
- `--delay 0.3` - Delay captions by seconds to avoid spoiling speech (default: 0.3s)
- `--max-words 4` - Maximum words per caption group (default: 4)

### Recommended Command

For best results, use all flags:

```bash
source .venv/bin/activate
python generate_captions.py input.m4a output.srt --model large-v3 --language en --vad --delay 0.3 --max-words 4
```

### Example

```bash
source .venv/bin/activate
python generate_captions.py 1003-1.m4a 1003-1.srt --model large-v3 --language en --vad --delay 0.3
```

**First run note**: The script will download the Whisper model (~3GB for large-v3). This only happens once—subsequent runs are offline.

## 🎬 Importing into CapCut

1. **Install the Lexend Deca font**:

   - Download from [Google Fonts](https://fonts.google.com/specimen/Lexend+Deca)
   - Install on your Mac/PC so CapCut can use it

2. **Import your video**:

   - Open CapCut
   - Import your video file (the same audio source you transcribed)

3. **Add captions**:

   - Go to **Text** → **Captions**
   - Click **Import subtitle** (or **Auto captions** → **Import**)
   - Select your generated `.srt` file (`1003-1.srt`)

4. **Adjust if needed**:
   - CapCut will apply the styling automatically
   - Preview and adjust timing if necessary
   - The colors, fonts, and emphasis should all be preserved

## 🛠 Technical Details

### How It Works

1. **Audio Conversion**: Uses FFmpeg (or fallback loaders) to obtain 16kHz mono audio
2. **Transcription**: Faster-Whisper produces segments with start/end times
3. **Sentence + Phrase Split**: Sentences broken at question marks, then phrases of up to 4 words
4. **Standalone Emphasis**: Emphasis words (e.g. SAVE, WIN, BONUS, LODAVO) become single-word captions
5. **Emoji Enhancement**: Alternating emoji addition after emphasis words, with cooldown spacing
6. **Comma Heuristics**: Intro words like LOOK, HEY, SO gain a trailing comma for natural pacing
7. **SRT Output**: Clean, uppercase, question-mark-only punctuation plus strategic commas

### Environment Setup

The project uses a minimal, clean environment:

**Core Dependencies**:

- `faster-whisper` - AI transcription engine
- `soundfile` - Audio file reading
- `numpy` - Array operations
- `ctranslate2` - Optimized inference
- `huggingface-hub` - Model downloading
- `tokenizers` - Text tokenization
- `onnxruntime` - Inference runtime

**System Dependencies**:

- FFmpeg (via Homebrew) - Audio conversion
- Python 3.13 (via Homebrew)

### Why No PyAV?

Building PyAV on macOS with Python 3.13 is problematic due to FFmpeg API incompatibilities. This script uses a workaround:

- A minimal `av` stub allows `faster-whisper` to import
- Audio loading uses FFmpeg + soundfile instead
- This approach is cleaner and more maintainable

## 🎨 Customization

### Modify Styles

ASS styling removed in favor of simplified SRT workflow. Styling (font, size, colors) should now be applied inside CapCut after import.

### Add More Emojis

Edit the `EMOJI_MAP` dictionary in `generate_captions.py`:

```python
EMOJI_MAP = {
    "save": "💰",
    "win": "🏆",
    # Add your own...
}
```

### Change Emphasis Keywords

Edit the `EMPHASIS_WORDS` set in `generate_captions.py` to control which words can stand alone and get emojis.

## 📋 Tips & Best Practices

1. **Model Selection**:

   - Use `large-v3` for best results (recommended, ~3GB download)
   - Use `small` for faster processing if needed (~500MB download)

2. **Audio Quality**:

   - Clear audio = better transcription
   - Remove background noise if possible
   - Consistent volume levels help

3. **Caption Timing**:

   - The script uses VAD to filter silence
   - Word timestamps enable smooth animations in CapCut
   - Adjust `max_chars` and `max_gap` in `natural_group()` for different pacing

4. **Font Installation**:

   - **Critical**: Install Lexend Deca before importing into CapCut
   - Without it, CapCut will substitute a different font

5. **Color Verification**:
   - Test in CapCut to ensure colors render correctly
   - Some platforms may interpret ASS colors differently

## 🐛 Troubleshooting

### "Model not found" or slow download

- First run downloads the model (~500MB for small)
- Ensure stable internet connection
- Models are cached in `~/.cache/huggingface/`

### "Format not recognised" error

- Install FFmpeg: `brew install ffmpeg`
- Check that FFmpeg is in your PATH: `which ffmpeg`

### Captions don't look right in CapCut

- Install the Lexend Deca font on your system
- Restart CapCut after installing fonts
- Re-import the .srt file

### Python version issues

- This project requires Python 3.13+
- Install via Homebrew: `brew install python@3.13`

## 📦 File Structure

```
lodavo-caption-generator/
├── generate_captions.py      # Main script (SRT output)
├── requirements_local.txt    # Python dependencies
├── inputs/                   # Audio sources
├── outputs/                  # Generated .srt files
├── README.md                 # This file
└── lodavo-caption-generator.code-workspace # VS Code workspace (git-ignored)
```

## 🔄 Workflow Summary

1. Record/export your audio → `video.m4a`
2. Run transcription → `python generate_captions.py video.m4a outputs/video.srt --model large-v3 --language en --vad --delay 0.3`
3. Import video into CapCut
4. Import generated `video.srt` subtitle file
5. Edit and export your viral TikTok! 🎉

## 📚 Resources

- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Whisper Model Info](https://github.com/openai/whisper)
- [Lexend Deca Font](https://fonts.google.com/specimen/Lexend+Deca)
- [CapCut Official Site](https://www.capcut.com/)

---

**Made with ❤️ for Lodavo** 💙
