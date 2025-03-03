import math
import cv2
import click
import sys
from tqdm import tqdm
import whisper
from moviepy.editor import VideoFileClip
from pathlib import Path
import json

from compare import compare_images
from images2pdf import images2pdf

# Load configuration from JSON file
with open('config.json', 'r') as f:
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
                export_pdf=False):
        self.url = Path(url)
        # Create output directory with video name
        self.output_path = Path(output_path) / self.url.stem
        self.images_path = self.output_path / 'images'
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
        
    def process(self):
        # Create output directories if they don't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(exist_ok=True)
        print(f"Output will be saved to: {self.output_path}")
        
        self._process_video()
        if self.export_pdf:
            self._export_pdf()
        self._process_audio()

    def _process_video(self):
        vcap = cv2.VideoCapture(str(self.url))  
        fps = int(vcap.get(5))
        total_frames = int(vcap.get(7))
        
        # Calculate frame sampling interval
        sample_interval = max(1, fps // self.fps_sample)

        if total_frames == 0:
            raise VideoProcessingError('Please check if the video url is correct')

        self.frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.start_time > total_frames / fps:
            raise VideoProcessingError('Start Time >= Video Duration')
        
        # set start frame
        vcap.set(cv2.CAP_PROP_POS_FRAMES, self.start_time * fps)
        frame_count = ((int(total_frames / fps) if self.end_time == float('inf') else self.end_time) - self.start_time) * fps
        
        # Calculate expected number of frames to process based on sampling interval
        total_samples = frame_count // sample_interval

        last_frame = []
        read_frames = 0
        similarity = 0
        
        with tqdm(total=total_samples, desc="Processing frames") as pbar:
            while True:
                ret, frame = vcap.read()
                if not ret or read_frames >= frame_count:
                    break
                
                read_frames += 1
                if read_frames % sample_interval != 0:
                    continue

                is_write = False

                if len(last_frame):
                    similarity = round(compare_images(frame, last_frame), 2)
                    if self.compare_method in ['dhash', 'phash']:
                        # For hash methods, lower score means more similar
                        is_write = similarity > self.similarity_threshold
                    else:
                        # For histogram, higher score means more similar
                        is_write = similarity < self.similarity_threshold
                else:
                    is_write = True

                if is_write:
                    name = self.images_path / f'frame{second2hms(math.ceil((read_frames + self.start_time * fps) / fps))}-{str(similarity)}.jpg'
                    if not cv2.imwrite(str(name), frame):
                        raise VideoProcessingError('Failed to write image file')

                    last_frame = frame
                
                pbar.update(1)
                if self._progress_callback:
                    self._progress_callback(read_frames / frame_count, "Processing frames...")

        vcap.release()
        cv2.destroyAllWindows()

    def _process_audio(self):
        """Extract audio and transcribe it using asr, default whisper"""
        try:
            if self._progress_callback:
                self._progress_callback(0, "Extracting audio...")
                
            # Extract audio
            video = VideoFileClip(str(self.url))  
            audio_path = self.output_path / config['OUTPUT_AUDIO_NAME']
            video.audio.write_audiofile(str(audio_path))
            video.close()

            if self._progress_callback:
                self._progress_callback(0.3, "Loading ASR model...")

            # Determine device
            if self.asr_device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.asr_device
            
            print(f"Using device: {device}")

            # Load ASR model and transcribe
            if self.asr_model.startswith('whisper-'):
                # Extract whisper model name after 'whisper-' prefix
                whisper_model_name = self.asr_model[8:]  # Remove 'whisper-' prefix
                print(f"Loading whisper model: {whisper_model_name} on {device}")
                model = whisper.load_model(whisper_model_name, device=device)
                print("Model loaded successfully")
                
                if self._progress_callback:
                    self._progress_callback(0.5, "Transcribing audio... this may take a little bit longer")
                    
                print("Transcribing audio...")
                result = model.transcribe(str(audio_path), fp16=(device == "cuda"))
                transcript_text = result["text"]
                print("Transcription completed")
            else:
                raise ValueError(f"Unsupported ASR model: {self.asr_model}")
            
            # Save transcript
            transcript_path = self.output_path / config['OUTPUT_TRANSCRIPT_NAME']
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
                
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
@click.argument('url', type=click.Path(exists=True, file_okay=True, dir_okay=False))
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

def main(similarity_threshold, pdfname, start_time, end_time, fps_sample, 
         asr_model, outputpath, asr_device, url, compare_method, export_pdf):
    """Process video file at URL into frames and transcript.
    
    URL: Path to the video file to process
    """
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
    
    try:
        processor.process()
        click.echo("Processing completed successfully!")
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