# import streamlit as st
# import cv2
# import librosa
# import numpy as np
# from processing import process_facial_emotions, process_speech_emotions, process_live_frame, emotions
# from essay_processing import process_essay, get_essay_feedback
# from feedback import get_facial_feedback_personalized, get_speech_feedback_personalized, get_cumulative_feedback
# from analytics import plot_emotion_trend, plot_emotion_pie, calculate_readiness_score, calculate_emotion_consistency, analyze_emotion_transitions, analyze_peak_emotions, analyze_emotion_duration, generate_pdf_report
# import tempfile
# import os
# import io
# import traceback
# from moviepy.editor import VideoFileClip
# import matplotlib.pyplot as plt
# import random
# import time

# def cleanup_file(file_path, max_retries=15, delay=3):  
#     """Attempt to delete a file with retries to handle Windows file locking."""
#     try:
#         import psutil
#     except ImportError:
#         psutil = None

#     for attempt in range(max_retries):
#         try:
#             if os.path.exists(file_path):
#                 try:
#                     with open(file_path, 'rb') as f:
#                         pass
#                 except IOError:
#                     pass
#                 if psutil:
#                     for proc in psutil.process_iter(['pid', 'name', 'open_files']):
#                         try:
#                             for file_handle in proc.open_files():
#                                 if file_handle.path == os.path.abspath(file_path):
#                                     proc.kill()
#                         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#                             pass
#                 try:
#                     import msvcrt
#                     if os.path.exists(file_path):
#                         handle = msvcrt.open_osfhandle(msvcrt.get_osfhandle(os.open(file_path, os.O_RDONLY)), 0)
#                         msvcrt.close(handle)
#                 except (ImportError, OSError, IOError):
#                     pass
#                 os.unlink(file_path)
#                 print(f"Successfully deleted {file_path}")
#                 return True
#         except PermissionError as e:
#             print(f"Attempt {attempt + 1}/{max_retries} failed to delete {file_path}: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(delay)
#         except Exception as e:
#             print(f"Unexpected error deleting {file_path}: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(delay)
#     st.warning(f"Could not delete {file_path} after {max_retries} attempts.")
#     return False

# def plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions):
#     print("Generating key emotion summary plot...")
#     start_time = time.time()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     facial_top = sorted(facial_counts.items(), key=lambda x: x[1], reverse=True)[:3]
#     speech_top = sorted(speech_counts.items(), key=lambda x: x[1], reverse=True)[:3]
#     facial_emotions, facial_values = zip(*facial_top) if facial_top else ([], [])
#     speech_emotions, speech_values = zip(*speech_top) if speech_top else ([], [])
#     all_emotions = list(set(facial_emotions + speech_emotions)) or emotions[:3]
#     facial_data = {emo: 0 for emo in all_emotions}
#     speech_data = {emo: 0 for emo in all_emotions}
#     for emo, val in facial_top:
#         facial_data[emo] = val
#     for emo, val in speech_top:
#         speech_data[emo] = val
#     x = np.arange(len(all_emotions))
#     width = 0.35
#     ax.bar(x - width/2, [facial_data[emo] for emo in all_emotions], width, color='blue', alpha=0.7, label='Facial')
#     ax.bar(x + width/2, [speech_data[emo] for emo in all_emotions], width, color='green', alpha=0.7, label='Speech')
#     ax.text(0.95, 0.95, f'Readiness Score: {readiness_score:.2f}%', transform=ax.transAxes, 
#             bbox=dict(facecolor='white', alpha=0.8), fontsize=10, verticalalignment='top', horizontalalignment='right')
#     ax.set_title("Key Emotion Summary for Interview Readiness", fontsize=14)
#     ax.set_ylabel("Emotion Frequency", fontsize=12)
#     ax.set_xticks(x)
#     ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=10)
#     ax.legend(fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100)
#     buf.seek(0)
#     plt.close()
#     print(f"Key emotion summary plot generated in {time.time() - start_time:.2f} seconds")
#     return buf

# def get_essay_feedback_personalized(sentiment, confidence, coherence):
#     """
#     Generate personalized feedback based on sentiment and coherence scores.
#     """
#     print("Generating personalized essay feedback...")
#     try:
#         feedback = []
        
#         # Sentiment feedback
#         if sentiment == "Positive":
#             feedback.append("Your essay has a positive tone, which can be effective for questions about aspirations, strengths, and motivation.")
#         elif sentiment == "Negative":
#             feedback.append("Your essay has a negative tone. Consider reframing your response with more positive language, especially when discussing challenges.")
#         else:
#             feedback.append("Your essay has a neutral tone. For interview questions about passion or motivation, consider incorporating more positive language.")
            
#         # Confidence feedback
#         if confidence < 60:
#             feedback.append("The sentiment in your essay is somewhat ambiguous. Using more definitive language could strengthen your message.")
#         else:
#             feedback.append("You express your ideas with confidence, which is good for interview responses.")
            
#         # Coherence feedback
#         if coherence < 40:
#             feedback.append("Your essay could benefit from better structure and flow. Consider adding transition words and varying sentence length.")
#         elif coherence < 70:
#             feedback.append("Your essay has decent coherence but could be improved. Try connecting ideas more clearly with transition phrases.")
#         else:
#             feedback.append("Your essay has excellent structure and coherence, making it easy to follow your thoughts.")
            
#         # Overall advice
#         feedback.append("\nFor interview preparation:")
#         if coherence < 60 or confidence < 60:
#             feedback.append("- Practice articulating your thoughts more clearly and confidently.")
#         if sentiment == "Negative":
#             feedback.append("- Focus on framing challenges positively, highlighting solutions and growth.")
#         feedback.append("- Keep responses concise while maintaining completeness.")
        
#         return "\n".join(feedback)
#     except Exception as e:
#         print(f"Error in get_essay_feedback_personalized: {str(e)}")
#         return "Error generating personalized feedback."

# def main():
#     st.title("Dual Emotion & Essay Analysis")
#     st.write("Prepare for your interview with advanced emotion analysis, essay evaluation, and real-time feedback!")

#     essay_questions = [
#         "Why do you want this job?",
#         "What are your strengths and weaknesses?",
#         "Describe a challenging situation and how you handled it.",
#         "Where do you see yourself in five years?",
#         "Why should we hire you?"
#     ]

#     mode = st.sidebar.selectbox("Choose Mode", ["Audio/Facial Analysis", "Essay Evaluation", "Live Preview"])

#     if mode == "Audio/Facial Analysis":
#         uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

#         if uploaded_file is not None:
#             with st.spinner("Saving uploaded video..."):
#                 tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#                 tfile.write(uploaded_file.read())
#                 tfile.close()
#                 video_path = tfile.name

#             progress_bar = st.progress(0)
#             status_text = st.empty()

#             try:
#                 # Process facial emotions
#                 st.write("Processing facial emotions on GPU...")
#                 status_text.write("Step 1/3: Analyzing facial expressions...")
#                 start_time = time.time()
#                 processed_video_path, facial_frame_feedback, facial_overall, facial_counts, total_faces, facial_timeline, processed_frames = process_facial_emotions(video_path, sample_rate=2)
#                 print(f"Facial processing took {time.time() - start_time:.2f} seconds")
#                 total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
#                 facial_progress = min(processed_frames / total_frames, 1.0) * 0.5 if total_frames > 0 else 0.5
#                 progress_bar.progress(facial_progress)
#                 st.write(f"Facial processing completed: Processed {processed_frames} of {total_frames} frames")

#                 # Process speech emotions
#                 st.write("Processing speech emotions on GPU...")
#                 status_text.write("Step 2/3: Analyzing speech tones...")
#                 start_time = time.time()
#                 audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
#                 video_clip = VideoFileClip(video_path)
#                 video_clip.audio.write_audiofile(audio_temp.name, codec='pcm_s16le')
#                 video_clip.close()
#                 audio_path = audio_temp.name
#                 print(f"Extracted audio to {audio_path}")
#                 speech_frame_feedback, speech_overall, speech_counts, total_chunks, speech_timeline, processed_chunks = process_speech_emotions(audio_path, chunk_length=3.0)
#                 print(f"Speech processing took {time.time() - start_time:.2f} seconds")
#                 audio, _ = librosa.load(audio_path, sr=22050)
#                 total_possible_chunks = len(audio) // int(3.0 * 22050)
#                 speech_progress = 0.5 + min(processed_chunks / total_possible_chunks, 1.0) * 0.5 if total_possible_chunks > 0 else 1.0
#                 progress_bar.progress(speech_progress)
#                 st.write(f"Speech processing completed: Processed {processed_chunks} of {total_possible_chunks} chunks")
#                 st.write("Extracted Audio File:")
#                 st.audio(audio_path, format='audio/wav')

#                 # Analytics and feedback
#                 st.write("Generating analytics and visualizations...")
#                 status_text.write("Step 3/3: Analyzing data and generating reports...")
#                 try:
#                     start_time = time.time()
#                     readiness_score = calculate_readiness_score(facial_counts, total_faces, speech_counts, total_chunks)
#                     consistency = calculate_emotion_consistency(facial_counts, total_faces, speech_counts, total_chunks, emotions)
#                     facial_sampled = facial_timeline[::20] if len(facial_timeline) > 50 else facial_timeline
#                     speech_sampled = speech_timeline[::20] if len(speech_timeline) > 50 else speech_timeline
                    
#                     plot_trend_buf = plot_emotion_trend(facial_sampled, speech_sampled, emotions)
#                     facial_pie_buf = plot_emotion_pie(facial_counts, total_faces, "Facial Emotion Distribution")
#                     speech_pie_buf = plot_emotion_pie(speech_counts, total_chunks, "Speech Emotion Distribution")
#                     summary_buf = plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions)
#                     print(f"Analytics generation took {time.time() - start_time:.2f} seconds")

#                     # Display results
#                     st.write("Results:")
#                     st.write("---")
                    
#                     st.write("### Facial Emotion Analysis")
#                     st.write("Facial Overall Emotion Percentages:")
#                     for percentage in facial_overall:
#                         st.write(percentage)
#                     st.image(facial_pie_buf, caption="Facial Emotion Pie Chart", width=300)
#                     st.write("Your Facial Feedback:")
#                     st.write(get_facial_feedback_personalized(facial_counts, total_faces))
#                     with st.expander("View Frame-by-Frame Facial Feedback"):
#                         for feedback in facial_frame_feedback:
#                             st.write(feedback)

#                     st.write("### Speech Emotion Analysis")
#                     st.write("Speech Overall Emotion Percentages:")
#                     for percentage in speech_overall:
#                         st.write(percentage)
#                     st.image(speech_pie_buf, caption="Speech Emotion Pie Chart", width=300)
#                     st.write("Your Speech Feedback:")
#                     st.write(get_speech_feedback_personalized(speech_counts, total_chunks))
#                     with st.expander("View Chunk-by-Chunk Speech Feedback"):
#                         for feedback in speech_frame_feedback:
#                             st.write(feedback)

#                     st.write("### Combined Feedback (Face + Voice)")
#                     st.write(get_cumulative_feedback(facial_counts, total_faces, speech_counts, total_chunks))

#                     st.write(f"**Interview Readiness Score:** {readiness_score:.2f}% (Industry benchmark: ~50% for positive impression)")
#                     st.write(f"**Emotion Consistency:** {consistency:.2f}% (Higher is better for coherence)")
#                     st.image(plot_trend_buf, caption="Emotion Trend Over Time", width=600)
#                     st.image(summary_buf, caption="Key Emotion Summary and Readiness", width=800)

#                     st.write("### Quick Tips:")
#                     st.write("- Focus on **Neutral** or **Happy** emotions for a strong impression.")
#                     st.write(f"- Your readiness score of {readiness_score:.2f}% suggests {'strong preparation' if readiness_score >= 50 else 'areas for improvement'}.")

#                     # PDF Report
#                     st.write("Generating PDF report...")
#                     pdf_path = generate_pdf_report(facial_overall, speech_overall, 
#                                                   get_facial_feedback_personalized(facial_counts, total_faces),
#                                                   get_speech_feedback_personalized(speech_counts, total_chunks),
#                                                   get_cumulative_feedback(facial_counts, total_faces, speech_counts, total_chunks),
#                                                   plot_trend_buf)
#                     progress_bar.progress(1.0)
#                     st.write("PDF report generated")
#                     with open(pdf_path, 'rb') as pdf_file:
#                         st.download_button("Download Report", pdf_file, "interview_report.pdf")

#                 except Exception as analytics_error:
#                     st.error(f"Error generating analytics: {str(analytics_error)}")
#                     st.write(traceback.format_exc())
#                     raise

#             except Exception as e:
#                 st.error(f"Error processing video: {str(e)}")
#                 st.write(traceback.format_exc())

#             # Cleanup
#             cleanup_files = [video_path, processed_video_path if 'processed_video_path' in locals() else None,
#                             pdf_path if 'pdf_path' in locals() else None, audio_path if 'audio_path' in locals() else None]
#             for file_path in cleanup_files:
#                 if file_path and not cleanup_file(file_path):
#                     st.warning(f"Some files could not be deleted automatically.")

#     elif mode == "Essay Evaluation":
#         st.write("### Essay Evaluation")
#         question = random.choice(essay_questions)
#         st.write(f"**Question:** {question}")
#         essay_text = st.text_area("Paste your essay here:", height=200, key="essay_input")

#         if st.button("Submit Essay"):
#             with st.spinner("Analyzing essay..."):
#                 if essay_text:
#                     try:
#                         start_time = time.time()
#                         essay_sentiment, essay_conf, essay_coherence = process_essay(essay_text)
#                         print(f"Essay processing took {time.time() - start_time:.2f} seconds")
#                         essay_feedback = get_essay_feedback_personalized(essay_sentiment, essay_conf, essay_coherence)
                        
#                         # Display essay results
#                         st.write("### Essay Evaluation Results")
#                         st.write(f"**Sentiment:** {essay_sentiment}")
#                         st.write(f"**Confidence:** {essay_conf:.2f}%")
#                         st.write(f"**Coherence Score:** {essay_coherence:.2f}/100")
#                         st.write("**Essay Feedback:**")
#                         st.write(essay_feedback)
#                     except Exception as e:
#                         st.error(f"Error during essay processing: {str(e)}")
#                         st.write(traceback.format_exc())
#                 else:
#                     st.warning("Please enter an essay to evaluate.")

#     elif mode == "Live Preview":
#         st.write("Live Emotion Feedback (Press 'q' to quit)")
#         cap = cv2.VideoCapture(0)
#         frame_placeholder = st.empty()
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             processed_frame, emotions_detected = process_live_frame(frame)
#             frame_placeholder.image(processed_frame, channels="BGR")
#             if emotions_detected:
#                 st.write("Real-Time Feedback:", ", ".join([f"{emo} ({conf:.2f}%)" for emo, conf in emotions_detected]))
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cap.release()

# if __name__ == "__main__":
#     main()



import streamlit as st
import cv2
import librosa
import numpy as np
from processing import process_facial_emotions, process_speech_emotions, process_live_frame, emotions
from essay_processing import process_essay, get_essay_feedback
from feedback import get_facial_feedback_personalized, get_speech_feedback_personalized, get_cumulative_feedback, get_essay_feedback_personalized
from analytics import plot_emotion_trend, plot_emotion_pie, calculate_readiness_score, calculate_emotion_consistency, analyze_emotion_transitions, analyze_peak_emotions, analyze_emotion_duration, generate_pdf_report
import tempfile
import os
import io
import traceback
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import random
import time

def cleanup_file(file_path, max_retries=15, delay=3):  
    """Attempt to delete a file with retries to handle Windows file locking."""
    try:
        import psutil
    except ImportError:
        psutil = None

    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        pass
                except IOError:
                    pass
                if psutil:
                    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                        try:
                            for file_handle in proc.open_files():
                                if file_handle.path == os.path.abspath(file_path):
                                    proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                try:
                    import msvcrt
                    if os.path.exists(file_path):
                        handle = msvcrt.open_osfhandle(msvcrt.get_osfhandle(os.open(file_path, os.O_RDONLY)), 0)
                        msvcrt.close(handle)
                except (ImportError, OSError, IOError):
                    pass
                os.unlink(file_path)
                print(f"Successfully deleted {file_path}")
                return True
        except PermissionError as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed to delete {file_path}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error deleting {file_path}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    st.warning(f"Could not delete {file_path} after {max_retries} attempts.")
    return False

def plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions):
    print("Generating key emotion summary plot...")
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    facial_top = sorted(facial_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    speech_top = sorted(speech_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    facial_emotions, facial_values = zip(*facial_top) if facial_top else ([], [])
    speech_emotions, speech_values = zip(*speech_top) if speech_top else ([], [])
    all_emotions = list(set(facial_emotions + speech_emotions)) or emotions[:3]
    facial_data = {emo: 0 for emo in all_emotions}
    speech_data = {emo: 0 for emo in all_emotions}
    for emo, val in facial_top:
        facial_data[emo] = val
    for emo, val in speech_top:
        speech_data[emo] = val
    x = np.arange(len(all_emotions))
    width = 0.35
    ax.bar(x - width/2, [facial_data[emo] for emo in all_emotions], width, color='blue', alpha=0.7, label='Facial')
    ax.bar(x + width/2, [speech_data[emo] for emo in all_emotions], width, color='green', alpha=0.7, label='Speech')
    ax.text(0.95, 0.95, f'Readiness Score: {readiness_score:.2f}%', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8), fontsize=10, verticalalignment='top', horizontalalignment='right')
    ax.set_title("Key Emotion Summary for Interview Readiness", fontsize=14)
    ax.set_ylabel("Emotion Frequency", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_emotions, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    print(f"Key emotion summary plot generated in {time.time() - start_time:.2f} seconds")
    return buf

def get_essay_feedback_personalized(sentiment, confidence, coherence):
    """
    Generate personalized feedback based on sentiment and coherence scores.
    """
    print("Generating personalized essay feedback...")
    try:
        feedback = []
        
        # Sentiment feedback
        if sentiment == "Positive":
            feedback.append("Your essay has a positive tone, which can be effective for questions about aspirations, strengths, and motivation.")
        elif sentiment == "Negative":
            feedback.append("Your essay has a negative tone. Consider reframing your response with more positive language, especially when discussing challenges.")
        else:
            feedback.append("Your essay has a neutral tone. For interview questions about passion or motivation, consider incorporating more positive language.")
            
        # Confidence feedback
        if confidence < 60:
            feedback.append("The sentiment in your essay is somewhat ambiguous. Using more definitive language could strengthen your message.")
        else:
            feedback.append("You express your ideas with confidence, which is good for interview responses.")
            
        # Coherence feedback
        if coherence < 40:
            feedback.append("Your essay could benefit from better structure and flow. Consider adding transition words and varying sentence length.")
        elif coherence < 70:
            feedback.append("Your essay has decent coherence but could be improved. Try connecting ideas more clearly with transition phrases.")
        else:
            feedback.append("Your essay has excellent structure and coherence, making it easy to follow your thoughts.")
            
        # Overall advice
        feedback.append("\nFor interview preparation:")
        if coherence < 60 or confidence < 60:
            feedback.append("- Practice articulating your thoughts more clearly and confidently.")
        if sentiment == "Negative":
            feedback.append("- Focus on framing challenges positively, highlighting solutions and growth.")
        feedback.append("- Keep responses concise while maintaining completeness.")
        
        return "\n".join(feedback)
    except Exception as e:
        print(f"Error in get_essay_feedback_personalized: {str(e)}")
        return "Error generating personalized feedback."

def main():
    st.title("Complete Interview Preparation Analysis")
    st.write("Prepare for your interview with advanced emotion analysis, essay evaluation, and real-time feedback in one place!")

    essay_questions = [
        "Why do you want this job?",
        "What are your strengths and weaknesses?",
        "Describe a challenging situation and how you handled it.",
        "Where do you see yourself in five years?",
        "Why should we hire you?"
    ]

    mode = st.sidebar.selectbox("Choose Mode", ["Complete Analysis", "Live Preview"])

    if mode == "Complete Analysis":
        st.write("### Step 1: Upload Video")
        uploaded_file = st.file_uploader("Choose a video file for facial and speech analysis", type=["mp4", "avi", "mov"])
        
        st.write("### Step 2: Essay Evaluation")
        question = st.selectbox("Select an interview question:", essay_questions)
        st.write(f"**Question:** {question}")
        essay_text = st.text_area("Write or paste your essay answer here:", height=200, key="essay_input")
        
        analyze_button = st.button("Start Complete Analysis")

        if analyze_button and uploaded_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            with st.spinner("Processing video for facial and speech analysis..."):
                # Save uploaded video
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                video_path = tfile.name
                
                try:
                    # Process facial emotions
                    status_text.write("Step 1/4: Analyzing facial expressions...")
                    start_time = time.time()
                    processed_video_path, facial_frame_feedback, facial_overall, facial_counts, total_faces, facial_timeline, processed_frames = process_facial_emotions(video_path, sample_rate=2)
                    print(f"Facial processing took {time.time() - start_time:.2f} seconds")
                    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
                    facial_progress = min(processed_frames / total_frames, 1.0) * 0.3 if total_frames > 0 else 0.3
                    progress_bar.progress(facial_progress)
                    
                    # Process speech emotions
                    status_text.write("Step 2/4: Analyzing speech tones...")
                    start_time = time.time()
                    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    video_clip = VideoFileClip(video_path)
                    video_clip.audio.write_audiofile(audio_temp.name, codec='pcm_s16le')
                    video_clip.close()
                    audio_path = audio_temp.name
                    speech_frame_feedback, speech_overall, speech_counts, total_chunks, speech_timeline, processed_chunks = process_speech_emotions(audio_path, chunk_length=3.0)
                    print(f"Speech processing took {time.time() - start_time:.2f} seconds")
                    speech_progress = 0.3 + min(0.3, 0.3) # Update progress
                    progress_bar.progress(speech_progress)
                    
                    # Essay processing
                    status_text.write("Step 3/4: Analyzing essay content...")
                    if essay_text:
                        start_time = time.time()
                        essay_sentiment, essay_conf, essay_coherence = process_essay(essay_text)
                        essay_feedback = get_essay_feedback_personalized(essay_sentiment, essay_conf, essay_coherence)
                        print(f"Essay processing took {time.time() - start_time:.2f} seconds")
                    else:
                        essay_sentiment, essay_conf, essay_coherence = "Neutral", 0, 0
                        essay_feedback = "No essay was provided for analysis."
                    
                    progress_bar.progress(0.8)
                    
                    # Generate combined feedback
                    status_text.write("Step 4/4: Generating combined analysis and report...")
                    start_time = time.time()
                    readiness_score = calculate_readiness_score(facial_counts, total_faces, speech_counts, total_chunks)
                    consistency = calculate_emotion_consistency(facial_counts, total_faces, speech_counts, total_chunks, emotions)
                    facial_sampled = facial_timeline[::20] if len(facial_timeline) > 50 else facial_timeline
                    speech_sampled = speech_timeline[::20] if len(speech_timeline) > 50 else speech_timeline
                    
                    plot_trend_buf = plot_emotion_trend(facial_sampled, speech_sampled, emotions)
                    facial_pie_buf = plot_emotion_pie(facial_counts, total_faces, "Facial Emotion Distribution")
                    speech_pie_buf = plot_emotion_pie(speech_counts, total_chunks, "Speech Emotion Distribution")
                    summary_buf = plot_key_emotion_summary(facial_counts, speech_counts, readiness_score, emotions)
                    
                    # Generate combined feedback including essay
                    combined_feedback = get_cumulative_feedback(
                        facial_counts, total_faces, 
                        speech_counts, total_chunks,
                        essay_sentiment, essay_conf, essay_coherence
                    )
                    
                    print(f"Analytics generation took {time.time() - start_time:.2f} seconds")
                    progress_bar.progress(1.0)
                    
                    # Display all results
                    st.success("Analysis complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Facial Emotion Analysis")
                        st.write("Facial Overall Emotion Percentages:")
                        for percentage in facial_overall:
                            st.write(percentage)
                        st.image(facial_pie_buf, caption="Facial Emotion Pie Chart")
                        st.write("Your Facial Feedback:")
                        st.write(get_facial_feedback_personalized(facial_counts, total_faces))
                    
                    with col2:
                        st.write("### Speech Emotion Analysis")
                        st.write("Speech Overall Emotion Percentages:")
                        for percentage in speech_overall:
                            st.write(percentage)
                        st.image(speech_pie_buf, caption="Speech Emotion Pie Chart")
                        st.write("Your Speech Feedback:")
                        st.write(get_speech_feedback_personalized(speech_counts, total_chunks))
                    
                    st.write("### Essay Analysis")
                    st.write(f"**Question:** {question}")
                    st.write(f"**Sentiment:** {essay_sentiment}")
                    st.write(f"**Confidence:** {essay_conf:.2f}%")
                    st.write(f"**Coherence Score:** {essay_coherence:.2f}/100")
                    st.write("**Essay Feedback:**")
                    st.write(essay_feedback)
                    
                    st.write("### Combined Analysis")
                    st.write("#### Comprehensive Interview Readiness Feedback")
                    st.write(combined_feedback)
                    st.write(f"**Interview Readiness Score:** {readiness_score:.2f}% (Industry benchmark: ~50% for positive impression)")
                    st.write(f"**Emotion Consistency:** {consistency:.2f}% (Higher is better for coherence)")
                    st.image(plot_trend_buf, caption="Emotion Trend Over Time")
                    st.image(summary_buf, caption="Key Emotion Summary and Readiness")
                    
                    # Generate PDF report with combined feedback
                    st.write("Generating comprehensive PDF report...")
                    pdf_path = generate_pdf_report(
                        facial_overall, speech_overall, 
                        get_facial_feedback_personalized(facial_counts, total_faces),
                        get_speech_feedback_personalized(speech_counts, total_chunks),
                        combined_feedback,  # Using the combined feedback
                        plot_trend_buf,
                        essay_question=question,
                        essay_sentiment=essay_sentiment,
                        essay_conf=essay_conf,
                        essay_coherence=essay_coherence,
                        essay_feedback=essay_feedback
                    )
                    
                    with open(pdf_path, 'rb') as pdf_file:
                        st.download_button("Download Complete Interview Analysis Report", pdf_file, "interview_complete_report.pdf")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.write(traceback.format_exc())
                
                # Cleanup
                cleanup_files = [video_path, processed_video_path if 'processed_video_path' in locals() else None,
                                pdf_path if 'pdf_path' in locals() else None, audio_path if 'audio_path' in locals() else None]
                for file_path in cleanup_files:
                    if file_path and not cleanup_file(file_path):
                        st.warning(f"Some files could not be deleted automatically.")
        
        elif analyze_button and not uploaded_file:
            st.warning("Please upload a video file to proceed with the complete analysis.")

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