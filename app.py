import gradio as gr
import os
from pathlib import Path
from transmeet import VideoProcessor
import json
import time
import shutil
from images2pdf import images2pdf

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

def process_video(
    video_path,
    similarity_threshold=config['SIMILARITY_THRESHOLD'],
    start_time=config['START_TIME'],
    end_time=config['END_TIME'],
    fps_sample=config['FRAMES_PER_SECOND'],
    asr_model=config['ASR_MODEL'],
    asr_device=config['ASR_DEVICE'],
    compare_method=config['COMPARE_METHOD'],
    progress=gr.Progress()
):
    if not video_path:
        return "Please upload a video file.", None, None, None
    
    try:
        # Ensure base output directory exists
        output_dir = Path(config['OUTPUT_DIR'])
        output_dir.mkdir(exist_ok=True)
        
        # Create video-specific subdirectory
        video_name = Path(video_path).stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        processor = VideoProcessor(
            url=video_path,
            output_path=output_dir,
            similarity_threshold=float(similarity_threshold),
            start_time=start_time,
            end_time=end_time,
            fps_sample=int(fps_sample),
            asr_model=asr_model,
            asr_device=asr_device,
            compare_method=compare_method,
            export_pdf=False  # Don't export PDF during initial processing
        )
        
        # Create a progress tracker that maintains separate progress for each stage
        class StageProgress:
            def __init__(self, progress_callback):
                self.progress_callback = progress_callback
                self.current_stage = "Initializing"
                
            def update(self, progress_value, desc):
                self.current_stage = desc
                self.progress_callback(progress_value, desc=desc)
        
        stage_progress = StageProgress(progress)
        processor._progress_callback = stage_progress.update
        
        # Process the video
        processor.process()
        
        # Get the transcript path
        transcript_path = video_output_dir / config['OUTPUT_TRANSCRIPT_NAME']
        
        # Read the transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
            
        # Get all generated images
        images_path = video_output_dir / 'images'
        image_files = sorted(list(images_path.glob('*.jpg')))
        
        return [str(img) for img in image_files], transcript, str(video_output_dir), "Processing completed! Please click 'Results & Export' tab to view results."
        
    except Exception as e:
        return [], f"Error processing video: {str(e)}", None, None

def export_results(output_dir, export_pdf, export_audio, export_transcript):
    try:
        if not output_dir:
            return None, None, None, "Please process a video first."
        
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return None, None, None, "Output directory not found."
        
        result_files = []
        pdf_path = None
        audio_path = None
        transcript_path = None
        
        if export_pdf:
            images_path = output_dir / 'images'
            images = sorted(list(images_path.glob('*.jpg')))
            if images:
                pdf_path = output_dir / config['OUTPUT_PDF_NAME']
                # Use default frame size from config
                images2pdf(str(pdf_path), [str(img) for img in images], 
                         config['VIDEO_WIDTH'], config['VIDEO_HEIGHT'])
                result_files.append(f"PDF generated at {pdf_path}")
                pdf_path = str(pdf_path)
        
        if export_audio:
            audio_file = output_dir / config['OUTPUT_AUDIO_NAME']
            if audio_file.exists():
                audio_path = str(audio_file)
                result_files.append(f"Audio file ready at {audio_file}")
        
        if export_transcript:
            transcript_file = output_dir / config['OUTPUT_TRANSCRIPT_NAME']
            if transcript_file.exists():
                transcript_path = str(transcript_file)
                result_files.append(f"Transcript file ready at {transcript_file}")
        
        message = "No files selected for export." if not result_files else "\n".join(result_files)
        return pdf_path, audio_path, transcript_path, message
        
    except Exception as e:
        return None, None, None, f"Error during export: {str(e)}"

def reset_config():
    return (
        config['SIMILARITY_THRESHOLD'],
        config['FRAMES_PER_SECOND'],
        config['START_TIME'],
        config['END_TIME'],
        config['ASR_MODEL'],
        config['ASR_DEVICE'],
        config['COMPARE_METHOD']
    )

# Create the Gradio interface
with gr.Blocks(title="TransMeet - Video Processing Tool") as demo:
    gr.Markdown("""
    # TransMeet
    ## Video Processing and Transcription Tool
    """)
    
    with gr.Tabs() as tabs:
        # First tab for video processing
        with gr.Tab("Process Video"):
            gr.Markdown("Upload a video to extract key frames and generate transcript.")
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(label="Upload Video")
                    
                    with gr.Accordion("Configuration", open=False):
                        with gr.Row():
                            similarity_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=config['SIMILARITY_THRESHOLD'],
                                step=0.01,
                                label="Similarity Threshold"
                            )
                            fps_sample = gr.Number(
                                value=config['FRAMES_PER_SECOND'],
                                label="Frames Per Second",
                                precision=0
                            )
                        
                        with gr.Row():
                            start_time = gr.Textbox(
                                value=config['START_TIME'],
                                label="Start Time (HH:MM:SS)"
                            )
                            end_time = gr.Textbox(
                                value=config['END_TIME'],
                                label="End Time (HH:MM:SS)"
                            )
                        
                        with gr.Row():
                            asr_model = gr.Dropdown(
                                choices=["whisper-base", "whisper-large-v3-turbo", "whisper-large-v3-zh"],
                                value=config['ASR_MODEL'],
                                label="ASR Model"
                            )
                            asr_device = gr.Dropdown(
                                choices=["auto", "cuda", "cpu"],
                                value=config['ASR_DEVICE'],
                                label="ASR Device"
                            )
                        
                        compare_method = gr.Dropdown(
                            choices=["histogram", "dhash", "phash"],
                            value=config['COMPARE_METHOD'],
                            label="Frame Comparison Method"
                        )
                        
                        reset_btn = gr.Button("Reset to Default")
                    
                    process_btn = gr.Button("Process Video", variant="primary")
                    process_status = gr.Textbox(label="Status", interactive=False)
            
            output_dir = gr.State()
        
        # Second tab for results and export
        with gr.Tab("Results & Export", id="results_tab"):
            gr.Markdown("Review extracted frames and transcript, then choose what to export.")
            with gr.Row():
                with gr.Column():
                    output_gallery = gr.Gallery(label="Extracted Frames")
                    output_transcript = gr.Textbox(label="Transcript", lines=10)
                    with gr.Row():
                        export_pdf = gr.Checkbox(label="Export PDF", value=True)
                        export_audio = gr.Checkbox(label="Export Audio", value=False)
                        export_transcript = gr.Checkbox(label="Export Transcript", value=False)
                    export_btn = gr.Button("Export Selected Files")
                    export_result = gr.Textbox(label="Export Status", lines=3)
                    with gr.Row():
                        pdf_download = gr.File(label="Download PDF")
                        audio_download = gr.File(label="Download Audio")
                        transcript_download = gr.File(label="Download Transcript")
    
    # Connect the components
    reset_btn.click(
        fn=reset_config,
        inputs=[],
        outputs=[
            similarity_threshold,
            fps_sample,
            start_time,
            end_time,
            asr_model,
            asr_device,
            compare_method
        ]
    )
    
    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            similarity_threshold,
            start_time,
            end_time,
            fps_sample,
            asr_model,
            asr_device,
            compare_method
        ],
        outputs=[
            output_gallery,
            output_transcript,
            output_dir,
            process_status
        ]
    )
    
    export_btn.click(
        fn=export_results,
        inputs=[
            output_dir,
            export_pdf,
            export_audio,
            export_transcript
        ],
        outputs=[
            pdf_download,
            audio_download,
            transcript_download,
            export_result
        ]
    )

if __name__ == "__main__":
    demo.launch()