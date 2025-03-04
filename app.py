import gradio as gr
import os
from pathlib import Path
import sys
from omg import VideoProcessor
import json
import mss
from utils.images2pdf import images2pdf
import threading
import time
import queue

# Setup FFmpeg path
def get_ffmpeg_path():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle (exe)
        application_path = sys._MEIPASS
    else:
        # If the application is run from a Python interpreter
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    ffmpeg_path = os.path.join(application_path, 'ffmpeg', 'bin')
    if os.path.exists(ffmpeg_path):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Initialize FFmpeg path
get_ffmpeg_path()

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
        return [], "", None, "Please upload a video file or use screen capture."
    
    try:
        # Ensure base output directory exists
        output_dir = Path(config['OUTPUT_DIR'])
        output_dir.mkdir(exist_ok=True)
        
        processor = VideoProcessor(
            url=video_path,
            output_path=output_dir,
            similarity_threshold=float(similarity_threshold),
            start_time=start_time,
            end_time=end_time,
            fps_sample=fps_sample, 
            asr_model=asr_model,
            asr_device=asr_device,
            compare_method=compare_method,
            export_pdf=False
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
        
        # Get the transcript path if video was processed
        transcript = ""
        if video_path:
            transcript_path = processor.output_path / config['OUTPUT_TRANSCRIPT_NAME']
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            
        # Get all generated images
        images_path = processor.images_path
        image_files = sorted(list(images_path.glob('*.jpg')))
        
        return [str(img) for img in image_files], transcript, str(processor.output_path), "Processing completed! Please click 'Results & Export' tab to view results."
        
    except Exception as e:
        import traceback
        print(f"\nError processing video:\n{traceback.format_exc()}")
        return [], "", None, f"Error: {str(e)}"

def capture_screen(
    capture_type,
    monitor_selection,
    window_title,
    similarity_threshold=config['SIMILARITY_THRESHOLD'],
    fps_sample=config['FRAMES_PER_SECOND'],
    compare_method=config['COMPARE_METHOD'],
    progress=gr.Progress()
):
    try:
        # Get monitor index from selection if using monitor capture
        monitor_index = None
        if capture_type == "Monitor":
            with mss.mss() as sct:
                monitor_choices = {f"Monitor {i}: {m['width']}x{m['height']}": i 
                                for i, m in enumerate(sct.monitors)}
                monitor_index = monitor_choices[monitor_selection]
                if monitor_index >= len(sct.monitors):
                    raise ValueError(f"Monitor index {monitor_index} is not available")
        elif capture_type == "Window" and not window_title:
            raise ValueError("Please enter a window title for window capture mode")
        
        # Ensure base output directory exists
        output_dir = Path(config['OUTPUT_DIR'])
        output_dir.mkdir(exist_ok=True)
        
        processor = VideoProcessor(
            url=None,  # No video file for screen capture
            output_path=output_dir,
            similarity_threshold=float(similarity_threshold),
            fps_sample=fps_sample,
            compare_method=compare_method,
            export_pdf=False
        )
        
        # Create a queue for status updates
        status_queue = queue.Queue()
        
        # Create a progress tracker
        class StageProgress:
            def __init__(self, progress_callback):
                self.progress_callback = progress_callback
                self.current_stage = "Capturing screen"
                
            def update(self, progress_value, desc):
                self.current_stage = desc
                self.progress_callback(progress_value, desc=desc)
        
        stage_progress = StageProgress(progress)
        processor._progress_callback = stage_progress.update
        
        # Start screen capture in a separate thread
        def capture_thread():
            try:
                # Start screen capture based on capture type
                if capture_type == "Monitor":
                    processor.capture_screen(monitor=monitor_index)
                else:  # Window capture
                    processor.capture_screen(window_title=window_title)
            except Exception as e:
                error_msg = f"Error in capture thread: {str(e)}"
                print(error_msg)
                status_queue.put(("error", error_msg))
        
        capture_thread = threading.Thread(target=capture_thread, daemon=True)
        capture_thread.start()
        
        # Wait a short time to check for immediate errors
        time.sleep(0.5)
        try:
            # Check if there's an error message in the queue
            error_status = status_queue.get_nowait()
            if error_status[0] == "error":
                return [], "", None, error_status[1], None
        except queue.Empty:
            # No error, continue with capture
            return [], "", str(output_dir), "Screen capture started. Click 'Stop Capture' to finish.", processor
        
    except Exception as e:
        import traceback
        print(f"\nError in screen capture:\n{traceback.format_exc()}")
        return [], "", None, f"Error: {str(e)}", None

def export_results(output_dir, export_pdf, export_audio, export_transcript):
    try:
        if not output_dir:
            return None, None, None, "Please process a video first."
        
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return None, None, None, "Output directory not found."
        
        result_messages = []
        pdf_path = None
        audio_path = None
        transcript_path = None
        
        # For screen capture, we need to find the latest capture directory
        if output_dir.name == "screen_capture":
            # Find the latest timestamp directory
            capture_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            if capture_dirs:
                # Sort by directory name (timestamp) in descending order
                latest_dir = sorted(capture_dirs, reverse=True)[0]
                output_dir = latest_dir
        
        # Export PDF
        if export_pdf:
            try:
                images_path = output_dir / 'images'
                images = sorted(list(images_path.glob('*.jpg')))
                if images:
                    pdf_path = output_dir / config['OUTPUT_PDF_NAME']
                    # Use default frame size from config
                    images2pdf(str(pdf_path), [str(img) for img in images], 
                            config['VIDEO_WIDTH'], config['VIDEO_HEIGHT'])
                    result_messages.append(f"PDF generated at {pdf_path}")
                    pdf_path = str(pdf_path)
            except Exception as e:
                result_messages.append(f"PDF export failed: {str(e)}")
        
        # Export Audio
        if export_audio:
            try:
                audio_file = output_dir / config['OUTPUT_AUDIO_NAME']
                if audio_file.exists():
                    audio_path = str(audio_file)
                    result_messages.append(f"Audio file ready at {audio_file}")
                else:
                    result_messages.append("Audio file not found")
            except Exception as e:
                result_messages.append(f"Audio export failed: {str(e)}")
        
        # Export Transcript
        if export_transcript:
            try:
                transcript_file = output_dir / config['OUTPUT_TRANSCRIPT_NAME']
                if transcript_file.exists():
                    transcript_path = str(transcript_file)
                    result_messages.append(f"Transcript file ready at {transcript_file}")
                else:
                    result_messages.append("Transcript file not found")
            except Exception as e:
                result_messages.append(f"Transcript export failed: {str(e)}")
        
        message = "No files selected for export." if not result_messages else "\n".join(result_messages)
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

def delete_selected_images(selected_index, image_paths, output_directory):
    """Delete selected image and return updated image list and status"""
    if selected_index is None:
        return image_paths, "No image selected for deletion"
    
    try:
        # Convert paths to Path objects
        image_paths = [Path(p) if isinstance(p, str) else Path(p[0]) for p in image_paths]
        
        # Get the original image directory
        output_dir = Path(output_directory)
        images_dir = output_dir / 'images'
        original_images = sorted(list(images_dir.glob('*.jpg')))
        
        # Delete the selected image
        if 0 <= selected_index < len(original_images):
            image_path = original_images[selected_index]
            temp_path = image_paths[selected_index]
            if image_path.exists():
                image_path.unlink()
                print(f"Deleted image: {image_path}")
                status_msg = (
                    f"Successfully deleted image at position {selected_index + 1}\n"
                    f"Original path: {image_path}\n"
                    f"Temporary path: {temp_path}"
                )
        
        # Get updated list of images
        remaining_images = sorted(list(images_dir.glob('*.jpg')))
        return [str(img) for img in remaining_images], status_msg
    except Exception as e:
        import traceback
        print(f"Error in delete_selected_images:\n{traceback.format_exc()}")
        return image_paths, f"Error deleting image: {str(e)}"

def save_transcript(transcript_text, output_directory):
    """Save modified transcript and return status"""
    try:
        output_dir = Path(output_directory)
        transcript_path = output_dir / config['OUTPUT_TRANSCRIPT_NAME']
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
            
        return "Transcript saved successfully"
    except Exception as e:
        return f"Error saving transcript: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="TransMeet - Video Processing Tool") as demo:
    # Add global processor state
    processor_state = gr.State()
    
    gr.Markdown("""
    # TransMeet
    ## Online Meeting Processing and Transcription Tool
    """)
    
    with gr.Tabs() as tabs:
        # First tab for video processing and screen capture
        with gr.Tab("Process Video/Screen Capture"):
            gr.Markdown("Upload a video or capture your screen to extract key frames.")
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Video Upload"):
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
                                    precision=2,
                                    minimum=0.1,
                                    step=0.1
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
                                    choices=["whisper-base", "whisper-large-v3-turbo"],
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
                    
                    with gr.Tab("Screen Capture"):
                        # Get available monitors
                        with mss.mss() as sct:
                            monitor_choices = {f"Monitor {i}: {m['width']}x{m['height']}": i 
                                            for i, m in enumerate(sct.monitors)}
                        
                        capture_type = gr.Radio(
                            choices=["Monitor", "Window"],
                            value="Monitor",
                            label="Capture Type"
                        )
                        
                        monitor_select = gr.Dropdown(
                            choices=list(monitor_choices.keys()),
                            value=list(monitor_choices.keys())[0] if monitor_choices else None,
                            label="Select Monitor"
                        )
                        
                        window_title_input = gr.Textbox(
                            label="Window Title (or part of it)",
                            placeholder="Enter window title to capture"
                        )
                        
                        with gr.Accordion("Configuration", open=False):
                            with gr.Row():
                                similarity_threshold_capture = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=config['SIMILARITY_THRESHOLD'],
                                    step=0.01,
                                    label="Similarity Threshold"
                                )
                                fps_sample_capture = gr.Number(
                                    value=config['FRAMES_PER_SECOND'],
                                    label="Frames Per Second",
                                    precision=2,
                                    minimum=0.1,
                                    step=0.1
                                )
                            
                            with gr.Row():
                                asr_model_capture = gr.Dropdown(
                                    choices=["whisper-base", "whisper-large-v3-turbo"],
                                    value=config['ASR_MODEL'],
                                    label="ASR Model"
                                )
                                asr_device_capture = gr.Dropdown(
                                    choices=["auto", "cuda", "cpu"],
                                    value=config['ASR_DEVICE'],
                                    label="ASR Device"
                                )
                            
                            compare_method_capture = gr.Dropdown(
                                choices=["histogram", "dhash", "phash"],
                                value=config['COMPARE_METHOD'],
                                label="Frame Comparison Method"
                            )
                            
                            reset_btn_capture = gr.Button("Reset to Default")
                        
                        with gr.Row():
                            start_capture_btn = gr.Button("Start Capture", variant="primary")
                            stop_capture_btn = gr.Button("Stop Capture", variant="secondary")
                    
                    process_status = gr.Textbox(label="Status", interactive=False)
                    output_dir = gr.State()
            
        # Second tab for results and export
        with gr.Tab("Results & Export", id="results_tab"):
            gr.Markdown("Review and edit extracted frames and transcript, then choose what to export.")
            with gr.Row():
                with gr.Column():
                    # Make gallery interactive with multiple selection
                    output_gallery = gr.Gallery(
                        label="Extracted Frames",
                        show_label=True,
                        elem_id="gallery",
                        columns=[4],
                        allow_preview=True,
                        preview=True,
                        height="auto",
                        object_fit="contain",
                        interactive=True,
                        type="filepath",
                        selected_index=None  # Allow deselection
                    )

                    # Separate status displays for images and transcript
                    image_status = gr.Textbox(
                        label="Image Operation Status", 
                        interactive=False,
                        value="Ready"
                    )

                    delete_btn = gr.Button("Delete Selected Images", variant="secondary")
                    
                    # Make transcript editable
                    output_transcript = gr.Textbox(
                        label="Transcript", 
                        lines=10,
                        interactive=True
                    )
                    transcript_status = gr.Textbox(
                        label="Transcript Operation Status", 
                        interactive=False,
                        value="Ready"
                    )
                    
                    save_transcript_btn = gr.Button("Save Transcript", variant="secondary")
                    
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
    
    reset_btn_capture.click(
        fn=reset_config,
        inputs=[],
        outputs=[
            similarity_threshold_capture,
            fps_sample_capture,
            asr_model_capture,
            asr_device_capture,
            compare_method_capture
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
    
    def start_capture(*args):
        global current_processor
        try:
            # Create new processor instance
            result = capture_screen(*args)
            if len(result) == 5:  # Unpack the result
                images, transcript, output_dir, status, processor = result
                current_processor = processor
                return images, transcript, output_dir, status
            return None, None, None, "Failed to start capture"
        except Exception as e:
            return None, None, None, f"Error starting capture: {str(e)}"
    
    def stop_capture():
        global current_processor
        try:
            if current_processor and hasattr(current_processor, 'stop_capture'):
                # First stop the capture
                current_processor.stop_capture()
                
                # Get the results
                if current_processor.images_path and current_processor.images_path.exists():
                    image_files = sorted(list(current_processor.images_path.glob('*.jpg')))
                    
                    # Process audio transcription
                    audio_path = current_processor.images_path.parent / config['OUTPUT_AUDIO_NAME']
                    if audio_path.exists():
                        transcript = current_processor.transcribe_audio(
                            audio_path, 
                            current_processor.images_path.parent
                        )
                    else:
                        transcript = "No audio file found"
                    
                    return [str(img) for img in image_files], transcript, str(current_processor.images_path.parent), "Capture completed! Please click 'Results & Export' tab to view results."
                return None, None, None, "Capture stopped but no results were found."
            return None, None, None, "No active capture to stop."
        except Exception as e:
            return None, None, None, f"Error stopping capture: {str(e)}"
    
    start_capture_btn.click(
        fn=start_capture,
        inputs=[
            capture_type,
            monitor_select,
            window_title_input,
            similarity_threshold_capture,
            fps_sample_capture,
            compare_method_capture
        ],
        outputs=[
            output_gallery,
            output_transcript,
            output_dir,
            process_status
        ]
    )
    
    stop_capture_btn.click(
        fn=stop_capture,
        inputs=[],
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

    # Add this function to handle selection
    def update_selected(evt: gr.SelectData):
        """Handle gallery selection event"""
        return evt.index

    # In the Results & Export tab, after output_gallery definition:
    selected_index = gr.State(None)  # Store the selected index

    # Connect the selection event
    output_gallery.select(
        fn=update_selected,
        inputs=[],
        outputs=selected_index
    )

    # Update delete button click handler
    delete_btn.click(
        fn=delete_selected_images,
        inputs=[
            selected_index,
            output_gallery,
            output_dir
        ],
        outputs=[output_gallery, image_status]
    )
    
    save_transcript_btn.click(
        fn=save_transcript,
        inputs=[output_transcript, output_dir],
        outputs=[transcript_status]
    )

if __name__ == "__main__":
    # demo.launch(share=True)
    demo.launch()
