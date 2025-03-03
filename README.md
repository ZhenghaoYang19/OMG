# TransMeet

English | [简体中文](README_zh.md)

TransMeet is a tool for processing meeting recordings, extracting key frames, and generating transcripts. It features a user-friendly web interface built with Gradio.

## Features

- Extract key frames from video based on content similarity
- Generate transcripts using OpenAI's Whisper models
- Export results as PDF, audio, and text files
- User-friendly web interface
- Configurable processing parameters
- Support for multiple video formats

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transmeet.git
cd transmeet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

1. Start the web interface:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:7860`

3. Process a video:
   - Upload your video file
   - Adjust configuration settings (optional)
   - Click "Process Video"
   - Wait for processing to complete
   - Switch to "Results & Export" tab to view results

4. Export results:
   - Select desired export options (PDF/Audio/Transcript)
   - Click "Export Selected Files"
   - Download the exported files

## Configuration

You can adjust the following parameters:

- **Similarity Threshold** (0.0-1.0): Controls how different frames need to be to be considered key frames, suggested value between 0.5 and 0.8
- **Frames Per Second**: Number of frames to sample per second
- **Start/End Time**: Process only a specific portion of the video
- **ASR Model**: Choose between different Whisper models
- **ASR Device**: Select processing device (suggested: auto)
- **Frame Comparison Method**: Choose between different frame comparison algorithms

Default values can be modified in `config.json`.

## Project Structure

```
transmeet/
├── app.py              # Web interface
├── transmeet.py        # Core processing logic
├── compare.py          # Frame comparison functions
├── images2pdf.py       # PDF generation
├── config.json         # Configuration file
├── requirements.txt    # Python dependencies
└── output/            # Processing results
```

## Output Directory Structure

```
output/
└── video_name/
    ├── images/        # Extracted frames
    ├── audio.wav      # Extracted audio
    ├── transcript.txt # Generated transcript
    └── slides.pdf     # Generated PDF (optional)
```

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [wudududu/extract-video-ppt](https://github.com/wudududu/extract-video-ppt/tree/master) for the inspiration and reference
- [Gradio](https://www.gradio.app/) for the web interface
- [OpenAI](https://openai.com/) for the Whisper ASR model

