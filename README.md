# OMG (Online Meeting Guru)

English | [简体中文](README_zh.md)

OMG is a tool for processing meeting recordings (or screen captures), extracting key frames, and generating transcripts. It features a lightweight modern interface built with Gradio.

## Features

- Two input modes:
  - Video file upload: Process pre-recorded videos
  - Screen capture: Capture and process screen content in real-time
- Extract key frames based on content similarity
- Generate transcripts using OpenAI's Whisper models
- Support manual editing of extracted frames and transcript before export
- Export results as PDF, audio, and text files
- Lightweight modern interface built with Gradio
- Extreme compression, successfully reducing a 2.48GB video to just 321MB

## Requirements

- Python>=3.8 (tested on 3.12) 
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ZhenghaoYang19/omg.git
cd omg
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

3. Choose your input mode:

   ### Video Upload
   - Upload your video file
   - Adjust configuration settings (optional)
   - Click "Process Video"
   - Wait for processing to complete
   - Switch to "Results & Export" tab to view results

   ### Screen Capture
   - Select capture type (Monitor or Window)
   - For Monitor capture: Choose the monitor from the dropdown
   - For Window capture: Enter the window title (or part of it)(e.g. chrome, edge, TencentMeeting)
   - Adjust configuration settings (optional)
   - Click "Start Capture"
   - Click "Stop Capture" when finished
   - Switch to "Results & Export" tab to view results

4. Export results:
   - Select desired export options (PDF/Audio/Transcript)
   - Click "Export Selected Files"
   - Download the exported files

## Configuration

You can adjust the following parameters:

- **Similarity Threshold** (0.0-1.0): Controls how different frames need to be to be considered key frames, suggested value between 0.6 and 0.7
- **Frames Per Second**: Number of frames to sample per second, suggested value between 0.2 and 5
- **Start/End Time** (Video upload only): Process only a specific portion of the video
- **ASR Model**: Choose between different Whisper models
- **ASR Device**: Select processing device (suggested: auto)
- **Frame Comparison Method**: Choose between different frame comparison algorithms

Default values can be modified in `config.json`.

## Project Structure

```
omg/
├── app.py              # Web interface
├── omg.py        # Core processing logic
├── utils/
│   ├── compare.py      # Frame comparison functions
│   └── images2pdf.py   # PDF generation
├── config.json         # Configuration file
├── requirements.txt    # Python dependencies
└── output/            # Processing results
```

## Output Directory Structure

For video upload:
```
output/
└── video_name/
    ├── images/        # Extracted frames
    ├── audio.wav      # Extracted audio
    ├── transcript.txt # Generated transcript
    └── slides.pdf     # Generated PDF (optional)
```

For screen capture:
```
output/
└── screen_capture/
    └── YYYYMMDD_HHMMSS/  # Timestamp of capture
        ├── images/        # Captured frames
        ├── audio.wav      # Recorded audio
        ├── transcript.txt # Generated transcript
        └── slides.pdf     # Generated PDF (optional)
```

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [wudududu/extract-video-ppt](https://github.com/wudududu/extract-video-ppt/tree/master) for the inspiration and reference
- [Gradio](https://www.gradio.app/) for the web interface
- [OpenAI](https://openai.com/) for the Whisper ASR model


