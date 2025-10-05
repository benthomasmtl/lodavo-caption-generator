# Lodavo Caption Generator 💙

AI-powered caption generation tool built specifically for **Lodavo's video content creation workflow**. Creates engaging, accurate captions for short-form video content with perfect timing and CapCut optimization.

Built with OpenAI's Whisper for transcription and fine-tuned for Lodavo's branding and style requirements.

## ✨ Features

- **🎯 High Accuracy**: Uses Whisper large-v3 model with optimized settings
- **⚡ Smart Timing**: Configurable delay to prevent caption spoiling
- **🎨 Lodavo Branding**: Automatic spelling correction and brand consistency
- **😀 Smart Emojis**: Context-aware emoji placement with frequency control
- **📱 CapCut Ready**: SRT format optimized for direct CapCut import
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

```bash
python make_ass_from_audio_local.py input_audio.m4a output.ass --model small --language en --vad
```

### Parameters

- `input_audio.m4a` - Your audio/video file (.m4a, .mp3, .wav, .mov, etc.)
- `output.ass` - Output caption file name
- `--model small` - Whisper model size (tiny/base/small/medium/large-v3)
  - **small**: Fast, good accuracy (~500MB download, recommended)
  - **large-v3**: Best accuracy but slow (~3GB download)
- `--language en` - Language code (en for English)
- `--vad` - Enable Voice Activity Detection for cleaner segments

### Example

```bash
python make_ass_from_audio_local.py 1003-1.m4a 1003-1.ass --model small --language en --vad
```

**First run note**: The script will download the Whisper model (~500MB for small). This only happens once—subsequent runs are offline.

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
   - Select your generated `.ass` file (`1003-1.ass`)

4. **Adjust if needed**:
   - CapCut will apply the styling automatically
   - Preview and adjust timing if necessary
   - The colors, fonts, and emphasis should all be preserved

## 🛠 Technical Details

### How It Works

1. **Audio Conversion**: Uses FFmpeg to convert any audio format to 16kHz mono WAV
2. **Transcription**: Faster-Whisper (optimized Whisper implementation) transcribes with word-level timestamps
3. **Intelligent Grouping**: Merges short segments into readable caption lines (max 64 chars, max 0.6s gaps)
4. **Smart Wrapping**: Breaks long lines into ~2 lines max (~36 chars per line)
5. **Emoji Enhancement**: Automatically adds relevant emojis based on keywords
6. **Emphasis Styling**: Bolds and enlarges brand-relevant words (save, win, tickets, bonus, Lodavo, etc.)
7. **Style Selection**: Chooses appropriate style (Default/Emphasis/GoldOnBlack) based on content

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

Edit the `build_ass_header()` function in `make_ass_from_audio_local.py`:

```python
Style: Default,Lexend Deca,64,&H00FFFFFF,&H00FFFFFF,&H00B30022,&H00000000,0,0,0,0,100,100,0,0,1,5,0,2,80,80,120,1
```

Format: `Name,Font,Size,PrimaryColor,SecondaryColor,OutlineColor,BackColor,...`

**Colors are in BGR format with `&H00` prefix**:

- Lodavo Blue `#2200B3` → `&H00B30022`
- Gold `#FFD700` → `&H0000D7FF`

### Add More Emojis

Edit the `EMOJI_MAP` dictionary:

```python
EMOJI_MAP = {
    "save": "💰",
    "win": "🏆",
    # Add your own...
}
```

### Change Emphasis Keywords

Edit the `KEYWORDS` list in `emphasize()`:

```python
KEYWORDS = [
    "save","win","prize","Lodavo",
    # Add more words to emphasize...
]
```

## 📋 Tips & Best Practices

1. **Model Selection**:

   - Use `small` for most cases (fast, good accuracy)
   - Use `large-v3` only if you need maximum accuracy and can wait

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
- Re-import the .ass file

### Python version issues

- This project requires Python 3.13+
- Install via Homebrew: `brew install python@3.13`

## 📦 File Structure

```
lodavo-captions/
├── make_ass_from_audio_local.py  # Main script
├── requirements_local.txt         # Python dependencies
├── 1003-1.m4a                    # Example audio input
├── 1003-1.ass                    # Generated caption file
├── README.md                      # This file
├── .venv/                         # Virtual environment (not in git)
└── lodavo-captions.code-workspace # VS Code workspace
```

## 🔄 Workflow Summary

1. Record/export your audio → `video.m4a`
2. Run transcription → `python make_ass_from_audio_local.py video.m4a video.ass --model small --language en --vad`
3. Import video into CapCut
4. Import generated `video.ass` subtitle file
5. Edit and export your viral TikTok! 🎉

## 📚 Resources

- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [Whisper Model Info](https://github.com/openai/whisper)
- [ASS Subtitle Format](http://www.tcax.org/docs/ass-specs.htm)
- [Lexend Deca Font](https://fonts.google.com/specimen/Lexend+Deca)
- [CapCut Official Site](https://www.capcut.com/)

---

**Made with ❤️ for Lodavo** 💙
