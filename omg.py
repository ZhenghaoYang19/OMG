import math
import cv2
import click
import sys
from tqdm import tqdm
import whisper
from pathlib import Path
import json
import win32gui
import win32con
import datetime
import threading
import queue
import sounddevice as sd
import soundfile as sf

from utils.compare_images import compare_images
from utils.images2pdf import images2pdf

# Load configuration from JSON file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class VideoNotFoundError(VideoProcessingError):
    """Raised when video file cannot be opened"""
    pass

class InvalidTimeFormatError(VideoProcessingError):
    """Raised when time format is invalid"""
    pass

class VideoProcessor:
    def __init__(self, url, output_path, similarity_threshold=config['SIMILARITY_THRESHOLD'], 
                pdf_name=config['OUTPUT_PDF_NAME'], start_time=config['START_TIME'], end_time=config['END_TIME'],
                fps_sample=config['FRAMES_PER_SECOND'], asr_model=config['ASR_MODEL'],
                asr_device=config['ASR_DEVICE'], compare_method=config['COMPARE_METHOD'],
                asr_prompt=config['ASR_PROMPT'],
                export_pdf=False):
        self.url = Path(url) if url else None
        # Create output directory with video name or screen capture
        self.output_path = Path(output_path)
        if self.url:
            # For video processing, use video name
            self.output_path = self.output_path / self.url.stem
        else:
            # For screen capture, use screen_capture as base folder
            self.output_path = self.output_path / "screen_capture"
        self.images_path = None  # Will be set during processing
        self.similarity_threshold = similarity_threshold
        self.pdf_name = pdf_name
        self.start_time = hms2second(start_time)
        self.end_time = hms2second(end_time)
        self.fps_sample = fps_sample
        self.frame_width = config['VIDEO_WIDTH']
        self.frame_height = config['VIDEO_HEIGHT']
        self.asr_model = asr_model
        assert asr_device in ['auto', 'cuda', 'cpu'], "Invalid ASR device, select from auto, cuda, cpu"
        self.asr_device = asr_device
        self.compare_method = compare_method
        self.export_pdf = export_pdf
        self._progress_callback = None
        self._stop_capture = False
        self._audio_recording = False
        self._audio_thread = None
        self._audio_queue = None
        self.asr_prompt = asr_prompt
        
    def process(self):
        # Create output directories if they don't exist
        
        if self.url:
            self._process_video()
        if self.export_pdf:
            self._export_pdf()
        if self.url:  # Only process audio if we have a video file
            self._process_audio()

    def _process_frames(self, frames_generator, total_frames=None):
        """Process frames from any source (video file or screen capture)"""
        # Ensure output directories exist and are writable
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(parents=True, exist_ok=True)
            print(f"Output will be saved to: {self.output_path}")
        except Exception as e:
            raise VideoProcessingError(f'Cannot create or write to output directory: {str(e)}')

        last_frame = []
        processed_frames = 0
        similarity = 0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame in frames_generator:
                if frame is None:
                    break
                
                processed_frames += 1
                is_write = False

                if len(last_frame):
                    similarity = round(compare_images(frame, last_frame, method=self.compare_method), 2)
                    if self.compare_method in ['dhash', 'phash']:
                        # For hash methods, lower score means more similar
                        is_write = similarity > self.similarity_threshold
                    else:
                        # For histogram, higher score means more similar
                        is_write = similarity < self.similarity_threshold
                else:
                    is_write = True

                if is_write:
                    timestamp = second2hms(processed_frames / self.fps_sample)
                    name = self.images_path / f'{timestamp}-{str(similarity)}.jpg'
                    success, img_encoded = cv2.imencode('.jpg', frame)
                    if success:
                        img_encoded.tofile(str(name))
                    else:
                        raise VideoProcessingError(f'Failed to write image file: {name}')
                    last_frame = frame
                
                if total_frames:
                    pbar.update(1)
                    if self._progress_callback:
                        self._progress_callback(processed_frames / total_frames, "Processing frames...")

    def _process_video(self):
        vcap = cv2.VideoCapture(str(self.url))  
        fps = vcap.get(5)  # Remove int() to keep original fps as float
        total_frames = int(vcap.get(7))
        self.images_path = self.output_path / 'images'
        
        # Calculate frame sampling interval in frames
        # For example, if fps=30 and self.fps_sample=0.5, interval would be 60 frames
        print(fps, self.fps_sample)
        sample_interval = round(fps / self.fps_sample)

        if total_frames == 0:
            raise VideoProcessingError('Please check if the video url is correct')

        self.frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.start_time > total_frames / fps:
            raise VideoProcessingError('Start Time >= Video Duration')
        
        # set start frame
        vcap.set(cv2.CAP_PROP_POS_FRAMES, self.start_time * fps)
        frame_count = ((int(total_frames / fps) if self.end_time == float('inf') else self.end_time) - self.start_time) * fps
        
        def frame_generator():
            read_frames = 0
            while True:
                ret, frame = vcap.read()
                if not ret or read_frames >= frame_count:
                    break
                
                read_frames += 1
                if read_frames % sample_interval != 0:
                    continue
                    
                yield frame
            
            vcap.release()
            cv2.destroyAllWindows()
        
        # Calculate expected number of frames to process based on sampling interval
        total_samples = math.ceil(frame_count / sample_interval)  # Use math.ceil for more accurate count
        self._process_frames(frame_generator(), total_samples)

    def transcribe_audio(self, audio_path, output_dir):
        """Transcribe audio file and save transcript"""
        try:
            print("\nProcessing audio with Whisper...")
            if self.asr_device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.asr_device
            
            model = whisper.load_model(self.asr_model[8:], device=device)  # Remove 'whisper-' prefix
            
            # Add prompt for better transcription
            prompt = self.asr_prompt
            result = model.transcribe(str(audio_path), fp16=(device == "cuda"), initial_prompt=prompt)
            
            # Save transcript with timestamps
            transcript_path = output_dir / config['OUTPUT_TRANSCRIPT_NAME']
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # Write each segment with its timestamp
                for segment in result['segments']:
                    start_time = str(datetime.timedelta(seconds=int(segment['start'])))
                    end_time = str(datetime.timedelta(seconds=int(segment['end'])))
                    f.write(f"[{start_time} -> {end_time}] {segment['text']}\n")
            print(f"Transcript with timestamps saved to: {transcript_path}")
            
            return result["text"]
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def stop_capture(self):
        """Stop screen capture and audio recording"""
        self._stop_capture = True
        if self._audio_recording:
            self._audio_recording = False
            if self._audio_thread:
                self._audio_thread.join()

    def capture_screen(self, monitor=0, window_title=None):
        """Capture screen or specific window in real-time"""
        import mss
        import numpy as np
        import time
        
        def get_window_info(title):
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title.lower() in window_title.lower():
                        windows.append((hwnd, window_title))
                return True
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            return windows

        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Audio status: {status}')
            self._audio_queue.put(indata.copy())
        
        with mss.mss() as sct:
            # Reset stop flag
            self._stop_capture = False
            
            # Create timestamped folder for this capture
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.images_path = self.output_path / timestamp / 'images'
            
            if window_title:
                # Find window by title
                windows = get_window_info(window_title)
                if not windows:
                    raise VideoProcessingError(f"No window found with title containing '{window_title}'")
                
                if len(windows) > 1:
                    print("\nMultiple windows found:")
                    for i, (_, title) in enumerate(windows):
                        print(f"{i}: {title}")
                    print("\nUsing the first window found.")
                
                hwnd = windows[0][0]
                window_name = windows[0][1]
                print(f"\nCapturing window: {window_name}")
                
                # Get window position and size
                try:
                    # Bring window to front
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(hwnd)
                    time.sleep(0.5)  # Give time for window to be brought to front
                    
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, x2, y2 = rect
                    monitor = {"top": y, "left": x, "width": x2-x, "height": y2-y}
                except Exception as e:
                    raise VideoProcessingError(f"Failed to get window position: {str(e)}")
            else:
                monitor = sct.monitors[monitor]
                print(f"\nCapturing full monitor: {monitor['width']}x{monitor['height']}")
            
            # Create output directories
            self.images_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving captures to: {self.images_path.parent}")

            # Initialize audio recording
            self._audio_queue = queue.Queue()
            sample_rate = 16000  # Hz
            channels = 1  # Mono
            audio_path = self.images_path.parent / config['OUTPUT_AUDIO_NAME']
            
            try:
                # Start audio recording in a separate thread
                print("\nStarting audio recording...")
                self._audio_recording = True
                audio_data = []
                
                with sf.SoundFile(str(audio_path), mode='w', samplerate=sample_rate,
                                channels=channels, format='WAV') as audio_file:
                    with sd.InputStream(samplerate=sample_rate, channels=channels,
                                    callback=audio_callback):
                        def audio_writer():
                            while self._audio_recording:
                                try:
                                    data = self._audio_queue.get(timeout=1)
                                    audio_file.write(data)
                                    audio_data.append(data)
                                except queue.Empty:
                                    continue
                        
                        self._audio_thread = threading.Thread(target=audio_writer)
                        self._audio_thread.start()
                        
                        def screen_generator():
                            last_capture_time = 0
                            capture_interval = 1.0 / self.fps_sample
                            
                            while not self._stop_capture:
                                current_time = time.time()
                                if current_time - last_capture_time >= capture_interval:
                                    # Capture screen/window
                                    screenshot = sct.grab(monitor)
                                    # Convert to numpy array
                                    frame = np.array(screenshot)
                                    # Convert from BGRA to BGR
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                                    yield frame
                                    last_capture_time = current_time
                                
                                # Small sleep to prevent excessive CPU usage
                                time.sleep(0.1)
                        
                        try:
                            self._process_frames(screen_generator())
                        finally:
                            # Stop audio recording
                            self._audio_recording = False
                            self._audio_thread.join()
                            
            except Exception as e:
                raise VideoProcessingError(f"Error during capture: {str(e)}")

    def _process_audio(self):
        """Extract audio and transcribe it using asr, default whisper"""
        try:
            if self._progress_callback:
                self._progress_callback(0, "Extracting audio...")
                
            # Extract audio using moviepy or ffmpeg directly
            try:
                from moviepy.editor import VideoFileClip
                # Try moviepy first
                video = VideoFileClip(str(self.url))  
                audio_path = self.output_path / config['OUTPUT_AUDIO_NAME']
                video.audio.write_audiofile(str(audio_path))
                video.close()
            except ImportError:
                # Fallback to ffmpeg directly if moviepy fails
                import subprocess
                # Use WAV format for better quality with Whisper
                audio_path = self.output_path / 'audio.wav'
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(self.url),
                        '-vn',  # Disable video
                        '-acodec', 'pcm_s16le',  # Audio codec (16-bit PCM)
                        '-ar', '16000',  # Sample rate (16kHz is ideal for Whisper)
                        '-ac', '1',  # Mono
                        '-y',  # Overwrite output file
                        str(audio_path)
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg stderr output: {e.stderr.decode()}")
                    raise

            if self._progress_callback:
                self._progress_callback(0.3, "Loading ASR model...First download needs some time")
            
            # Now call transcribe_audio to handle the transcription
            transcript_text = self.transcribe_audio(audio_path, self.output_path)
            
            # No need to save transcript again as transcribe_audio already does this
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise

    def _export_pdf(self):
        """Export images to PDF if requested"""
        images = list(self.images_path.glob('*.jpg'))
        images.sort()
        
        if not images:
            raise VideoProcessingError("No frames were extracted from the video")

        pdf_path = self.output_path / self.pdf_name
        images2pdf(str(pdf_path), [str(img) for img in images], self.frame_width, self.frame_height)

@click.command()
@click.argument('url', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False)
@click.option('--similarity_threshold', default=config['SIMILARITY_THRESHOLD'], 
              help='Similarity threshold for frame comparison (default: %(default)g)')
@click.option('--pdfname', default=config['OUTPUT_PDF_NAME'],
              help='Output PDF filename (default: %(default)s)')
@click.option('--start_time', default=config['START_TIME'],
              help='Start time (format: HH:MM:SS, default: %(default)s)')
@click.option('--end_time', default=config['END_TIME'],
              help='End time (format: HH:MM:SS, default: %(default)s)')
@click.option('--fps_sample', default=config['FRAMES_PER_SECOND'],
              help='Number of frames to sample per second (default: %(default)d)')
@click.option('--asr_model', default=config['ASR_MODEL'],
              help='ASR model to use for transcription (default: %(default)s)')
@click.option('--asr_device', default=config['ASR_DEVICE'],
              help='Device to use for ASR (auto/cuda/cpu) (default: %(default)s)')
@click.option('--outputpath', default=config['OUTPUT_DIR'],
              help='Output directory (default: %(default)s)')
@click.option('--compare_method', default=config['COMPARE_METHOD'],
              help='Comparison method for frame comparison (default: %(default)s)')
@click.option('--export_pdf', is_flag=True, default=False,
              help='Export frames to PDF (default: False)')
@click.option('--screen_capture', is_flag=True, default=False,
              help='Enable screen capture mode')
@click.option('--monitor', default=0,
              help='Monitor index for screen capture (default: 0)')
@click.option('--window_title', default=None,
              help='Window title (or part of it) to capture. If specified, --monitor is ignored.')
def main(similarity_threshold, pdfname, start_time, end_time, fps_sample, 
         asr_model, outputpath, asr_device, url, compare_method, export_pdf,
         screen_capture, monitor, window_title):
    """Process video file at URL into frames and transcript, or capture screen/window.
    
    URL: Optional path to the video file to process. Not needed for screen capture mode.
    """
    try:
        if screen_capture:
            if window_title:
                print(f"\nLooking for window with title containing: '{window_title}'")
            else:
                # List available monitors
                import mss
                with mss.mss() as sct:
                    print("\nAvailable monitors:")
                    for i, m in enumerate(sct.monitors):
                        print(f"Monitor {i}: {m['width']}x{m['height']}")
                    
                    if monitor >= len(sct.monitors):
                        click.echo(f"Error: Monitor index {monitor} is not available.", err=True)
                        sys.exit(1)
                    
                    print(f"\nUsing Monitor {monitor}: {sct.monitors[monitor]['width']}x{sct.monitors[monitor]['height']}")
            
            processor = VideoProcessor(
                url=None,
                output_path=outputpath,
                similarity_threshold=similarity_threshold,
                pdf_name=pdfname,
                start_time=start_time,
                end_time=end_time,
                fps_sample=fps_sample,
                asr_model=asr_model,
                asr_device=asr_device,
                compare_method=compare_method,
                export_pdf=export_pdf
            )
            
            print("\nStarting capture... Press Ctrl+C to stop.")
            processor.capture_screen(monitor=monitor, window_title=window_title)
            
        else:
            if not url:
                click.echo("Error: URL argument is required when not in screen capture mode.", err=True)
                sys.exit(1)
                
            processor = VideoProcessor(
                url=url,
                output_path=outputpath,
                similarity_threshold=similarity_threshold,
                pdf_name=pdfname,
                start_time=start_time,
                end_time=end_time,
                fps_sample=fps_sample,
                asr_model=asr_model,
                asr_device=asr_device,
                compare_method=compare_method,
                export_pdf=export_pdf
            )
            
            processor.process()
            
        click.echo("Processing completed successfully!")
        
    except KeyboardInterrupt:
        click.echo("\nScreen capture stopped by user.")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

def second2hms(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return ("%02d.%02d.%02d" % (h, m, s))

def hms2second(hms):
    if hms == config['END_TIME']:
        return float('inf')
    
    try:
        h, m, s = hms.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except ValueError:
        raise InvalidTimeFormatError(f"Invalid time format: {hms}. Expected format: HH:MM:SS")

if __name__ == '__main__':
    main()