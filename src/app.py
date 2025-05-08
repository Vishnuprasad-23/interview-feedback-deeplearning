import streamlit as st
import cv2
import librosa
import numpy as np
from processing import process_facial_emotions, process_speech_emotions, process_live_frame, emotions
from feedback import get_facial_feedback_personalized, get_speech_feedback_personalized, get_cumulative_feedback
from analytics import plot_emotion_trend, plot_emotion_pie, calculate_readiness_score, calculate_emotion_consistency, analyze_emotion_transitions, analyze_peak_emotions, analyze_emotion_duration, generate_pdf_report
import tempfile
import os
import io
import traceback
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

def cleanup_file(file_path, max_retries=15, delay=3):  
    """Attempt to delete a file with retries to handle Windows file locking, ensuring file is closed."""
    try:
        import psutil  # Optional Windows-specific module for checking open handles
    except ImportError:
        psutil = None  # Fallback if psutil is not installed

    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                # Try to close any open file handles (if possible)
                try:
                    with open(file_path, 'rb') as f:
                        pass  # Attempt to open and close to release locks
                except IOError:
                    pass  
                
                # Additional attempt to close handles using Windows-specific method (if psutil is available)
                if psutil:
                    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                        try:
                            for file_handle in proc.open_files():
                                if file_handle.path == os.path.abspath(file_path):
                                    print(f"Found open handle by process {proc.name} (PID: {proc.pid}) for {file_path}")
                                    proc.kill()  # Attempt to kill the process holding the file
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass  # Ignore if process can't be accessed or killed
                
                # Try using msvcrt for Windows-specific file handle manipulation
                try:
                    import msvcrt
                    if os.path.exists(file_path):
                        handle = msvcrt.open_osfhandle(msvcrt.get_osfhandle(os.open(file_path, os.O_RDONLY)), 0)
                        msvcrt.close(handle)
                except (ImportError, OSError, IOError):
                    pass  # Ignore if msvcrt fails (e.g., not Windows or file already closed)
                
                os.unlink(file_path)
                print(f"Successfully deleted {file_path}")
                return True
        except PermissionError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed to delete {file_path}: {e}")
            if attempt < max_retries - 1:
                # Use a simple counter-based delay without time module
                for _ in range(delay * 1000000):  # Busy wait for delay seconds
                    pass
        except Exception as e:
            print(f"Unexpected error deleting {file_path}: {e}")
            if attempt < max_retries - 1:
                for _ in range(delay * 1000000):  # Busy wait for delay seconds
                    pass
    st.warning(f"Could not delete {file_path} after {max_retries} attempts. Please delete manually from Temp folder.")
    return False

def plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions):
    """Create a simple bar chart summarizing dominant emotions and readiness for interviews."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get top 3 dominant emotions (by count/frequency) for facial and speech
    facial_top = sorted(facial_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    speech_top = sorted(speech_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Prepare data for plotting
    facial_emotions, facial_values = zip(*facial_top) if facial_top else ([], [])
    speech_emotions, speech_values = zip(*speech_top) if speech_top else ([], [])

    # Combine emotions and values, padding with zeros if necessary to match lengths
    all_emotions = list(set(facial_emotions + speech_emotions))
    if not all_emotions:
        all_emotions = emotions[:3]  # Default to top 3 emotions if none detected

    facial_data = {emo: 0 for emo in all_emotions}
    speech_data = {emo: 0 for emo in all_emotions}
    for emo, val in facial_top:
        facial_data[emo] = val
    for emo, val in speech_top:
        speech_data[emo] = val

    x = np.arange(len(all_emotions))
    width = 0.35  # Width of bars

    # Plot side-by-side bars for facial and speech emotions
    ax.bar(x - width/2, [facial_data[emo] for emo in all_emotions], width, color='blue', alpha=0.7, label='Facial')
    ax.bar(x + width/2, [speech_data[emo] for emo in all_emotions], width, color='green', alpha=0.7, label='Speech')

    # Add readiness score as a text box
    ax.text(0.95, 0.95, f'Readiness Score: {readiness_score:.2f}%', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8), fontsize=10, verticalalignment='top', horizontalalignment='right')

    # Customize plot
    ax.set_title("Key Emotion Summary for Interview Readiness", fontsize=14)
    ax.set_ylabel("Emotion Frequency", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)  # Lower DPI for faster rendering
    buf.seek(0)
    plt.close()
    return buf

def main():
    st.title("Dual Emotion Analysis")
    st.write("Prepare for your interview with advanced emotion analysis and real-time feedback!")

    # Sidebar for mode selection
    mode = st.sidebar.selectbox("Choose Mode", ["Upload Video", "Live Preview"])

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save uploaded video to temporary file
            with st.spinner("Saving uploaded video..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                video_path = tfile.name

            try:
                # Show real-time progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process facial emotions with progress updates (every frame)
                st.write("Processing facial emotions on GPU...")
                status_text.write("Step 1/3: Analyzing facial expressions...")
                processed_video_path, facial_frame_feedback, facial_overall, facial_counts, total_faces, facial_timeline, processed_frames = process_facial_emotions(video_path, sample_rate=1)  # Process every frame
                total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
                # Normalize progress to [0.0, 1.0] for 50% of the bar
                facial_progress = min(processed_frames / total_frames, 1.0) * 0.5 if total_frames > 0 else 0.5
                progress_bar.progress(facial_progress)
                st.write(f"Facial processing completed: Processed {processed_frames} of {total_frames} frames")

                # Process speech emotions with progress updates
                st.write("Processing speech emotions on GPU...")
                status_text.write("Step 2/3: Analyzing speech tones...")
                try:
                    # Extract audio to a .wav file for reliable librosa processing
                    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    video_clip = VideoFileClip(video_path)
                    video_clip.audio.write_audiofile(audio_temp.name, codec='pcm_s16le')
                    video_clip.close()
                    audio_path = audio_temp.name

                    print(f"Extracted audio to {audio_path}")  # Debug log

                    speech_frame_feedback, speech_overall, speech_counts, total_chunks, speech_timeline, processed_chunks = process_speech_emotions(audio_path, chunk_length=2.0)  # Use .wav file
                    # Estimate total possible chunks (using audio length from librosa)
                    audio, _ = librosa.load(audio_path, sr=22050)
                    total_possible_chunks = len(audio) // int(2.0 * 22050)  # 2.0 seconds per chunk at 22050 Hz
                    # Normalize progress to [0.0, 1.0] for the remaining 50% of the bar
                    speech_progress = 0.5 + min(processed_chunks / total_possible_chunks, 0.5) if total_possible_chunks > 0 else 1.0
                    progress_bar.progress(speech_progress)
                    st.write(f"Speech processing completed: Processed {processed_chunks} of {total_possible_chunks} chunks")

                    # Show extracted .wav file in UI
                    st.write("Extracted Audio File:")
                    st.audio(audio_path, format='audio/wav')

                except Exception as speech_error:
                    st.error(f"Error processing speech emotions: {str(speech_error)}")
                    st.write(traceback.format_exc())
                    raise

                # Analytics with progress (simplified, no timing)
                st.write("Generating analytics and visualizations...")
                status_text.write("Step 3/3: Analyzing data and generating reports...")
                try:
                    readiness_score = calculate_readiness_score(facial_counts, total_faces, speech_counts, total_chunks)
                    consistency = calculate_emotion_consistency(facial_counts, total_faces, speech_counts, total_chunks, emotions)
                    
                    # Sample timelines for faster plotting (e.g., every 10th point if > 100 points)
                    facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
                    speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline
                    
                    plot_trend_buf = plot_emotion_trend(facial_sampled, speech_sampled, emotions)
                    facial_pie_buf = plot_emotion_pie(facial_counts, total_faces, "Facial Emotion Distribution")
                    speech_pie_buf = plot_emotion_pie(speech_counts, total_chunks, "Speech Emotion Distribution")
                    progress_bar.progress(0.75)  # 75% for analytics
                    st.write(f"Analytics completed")

                except Exception as analytics_error:
                    st.error(f"Error generating analytics: {str(analytics_error)}")
                    st.write(traceback.format_exc())
                    raise

                # Display results
                st.write("Results:")
                st.write("---")
                
                st.write("Facial Overall Emotion Percentages:")
                for percentage in facial_overall:
                    st.write(percentage)
                st.image(facial_pie_buf, caption="Facial Emotion Pie Chart", width=300)

                st.write("Your Facial Feedback:")
                st.write(get_facial_feedback_personalized(facial_counts, total_faces))

                with st.expander("View Frame-by-Frame Facial Feedback"):
                    for feedback in facial_frame_feedback:
                        st.write(feedback)

                st.write("Speech Overall Emotion Percentages:")
                for percentage in speech_overall:
                    st.write(percentage)
                st.image(speech_pie_buf, caption="Speech Emotion Pie Chart", width=300)

                st.write("Your Speech Feedback:")
                st.write(get_speech_feedback_personalized(speech_counts, total_chunks))

                with st.expander("View Chunk-by-Chunk Speech Feedback"):
                    for feedback in speech_frame_feedback:
                        st.write(feedback)

                st.write("Combined Feedback (Face + Voice):")
                st.write(get_cumulative_feedback(facial_counts, total_faces, speech_counts, total_chunks))

                st.write(f"Interview Readiness Score: {readiness_score:.2f}% (Industry benchmark: ~50% for positive impression)")
                st.write(f"Emotion Consistency: {consistency:.2f}% (Higher is better for coherence)")
                st.image(plot_trend_buf, caption="Emotion Trend Over Time", width=600)  # Use st.image for matplotlib plot

                # Key Emotion Summary Visualization
                st.write("Key Insights for Interview Readiness:")
                summary_buf = plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions)
                st.image(summary_buf, caption="Key Emotion Summary and Readiness", width=800)

                # Optional concise text summary for quick reference
                st.write("Quick Tips:")
                st.write("- Focus on maintaining **Neutral** or **Happy** emotions for a positive interview impression.")
                st.write(f"- Your readiness score of {readiness_score:.2f}% suggests {'strong preparation' if readiness_score >= 50 else 'areas for improvement'} in emotional consistency.")

                # PDF Report with progress (simplified, no timing)
                st.write("Generating PDF report...")
                try:
                    pdf_path = generate_pdf_report(facial_overall, speech_overall, 
                                                  get_facial_feedback_personalized(facial_counts, total_faces),
                                                  get_speech_feedback_personalized(speech_counts, total_chunks),
                                                  get_cumulative_feedback(facial_counts, total_faces, speech_counts, total_chunks),
                                                  plot_trend_buf)
                    progress_bar.progress(1.0)  # 100% for PDF
                    st.write(f"PDF report generated")
                    with open(pdf_path, 'rb') as pdf_file:
                        st.download_button("Download Report", pdf_file, "interview_report.pdf")
                except Exception as pdf_error:
                    st.error(f"Error generating PDF report: {str(pdf_error)}")
                    st.write(traceback.format_exc())
                    raise

            except Exception as e:
                st.error(f"Error processing video: {e}")
                st.write(traceback.format_exc())

            # Cleanup with retry
            try:
                cleanup_files = [
                    video_path,
                    processed_video_path if 'processed_video_path' in locals() else None,
                    pdf_path if 'pdf_path' in locals() else None,
                    audio_path if 'audio_path' in locals() else None
                ]
                for file_path in cleanup_files:
                    if file_path and not cleanup_file(file_path):
                        st.warning(f"Some files could not be deleted automatically. Please check Temp folder.")
            except Exception as cleanup_error:
                st.warning(f"Cleanup error: {str(cleanup_error)}. Please delete temporary files manually from Temp folder.")

    elif mode == "Live Preview":
        st.write("Live Emotion Feedback (Press 'q' to quit)")
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, emotions_detected = process_live_frame(frame)
            frame_placeholder.image(processed_frame, channels="BGR")
            if emotions_detected:
                st.write("Real-Time Feedback:", ", ".join([f"{emo} ({conf:.2f}%)" for emo, conf in emotions_detected]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

if __name__ == "__main__":
    main()