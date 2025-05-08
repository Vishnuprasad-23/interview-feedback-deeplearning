# import matplotlib.pyplot as plt
# import numpy as np
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# from reportlab.lib.styles import getSampleStyleSheet
# import io
# import tempfile
# from collections import Counter

# def plot_emotion_trend(facial_timeline, speech_timeline, emotions, threshold=50.0):
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Simple layout

#     # Colors for emotions
#     colors = {
#         'Angry': 'red', 'Disgust': 'cyan', 'Fear': 'blue',
#         'Happy': 'green', 'Sad': 'yellow', 'Surprise': 'orange', 'Neutral': 'purple'
#     }

#     # Facial emotions (simplified to max confidence per frame)
#     facial_frames = [t[0] for t in facial_timeline]
#     facial_emotions = [t[1] for t in facial_timeline]
#     facial_confidences = [t[2] for t in facial_timeline]
#     for emotion in emotions:
#         y = [conf if emo == emotion and conf >= threshold else 0 for emo, conf in zip(facial_emotions, facial_confidences)]
#         ax1.plot(facial_frames, y, label=f'Facial - {emotion}', color=colors[emotion], linewidth=1, alpha=0.7)
#     ax1.set_title("Facial Emotion Trend (Confidence > 50%)", fontsize=12)
#     ax1.set_xlabel("Time (frames)", fontsize=10)
#     ax1.set_ylabel("Confidence (%)", fontsize=10)
#     ax1.legend(fontsize=8)
#     ax1.grid(True, linestyle='--', alpha=0.7)

#     # Speech emotions (simplified to max confidence per chunk)
#     speech_times = [t[0] for t in speech_timeline]
#     speech_emotions = [t[1] for t in speech_timeline]
#     speech_confidences = [t[2] for t in speech_timeline]
#     for emotion in emotions:
#         y = [conf if emo == emotion and conf >= threshold else 0 for emo, conf in zip(speech_emotions, speech_confidences)]
#         ax2.plot(speech_times, y, label=f'Speech - {emotion}', color=colors[emotion], linewidth=1, linestyle='--', alpha=0.7)
#     ax2.set_title("Speech Emotion Trend (Confidence > 50%)", fontsize=12)
#     ax2.set_xlabel("Time (seconds)", fontsize=10)
#     ax2.set_ylabel("Confidence (%)", fontsize=10)
#     ax2.legend(fontsize=8)
#     ax2.grid(True, linestyle='--', alpha=0.7)

#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100)  # Lower DPI for faster rendering
#     buf.seek(0)
#     plt.close()
#     return buf

# def plot_emotion_pie(counts, total, title):
#     labels = [emo for emo in counts.keys() if counts[emo] > 0]
#     sizes = [counts[emo] / total * 100 for emo in labels if counts[emo] > 0]
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
#     ax.axis('equal')
#     ax.set_title(title, fontsize=10)
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100)
#     buf.seek(0)
#     plt.close()
#     return buf

# def calculate_readiness_score(facial_counts, total_faces, speech_counts, total_chunks):
#     positive_emotions = ["Happy", "Neutral"]
#     facial_positive = sum(facial_counts[emo] for emo in positive_emotions) / total_faces if total_faces > 0 else 0
#     speech_positive = sum(speech_counts[emo] for emo in positive_emotions) / total_chunks if total_chunks > 0 else 0
#     score = (facial_positive + speech_positive) / 2 * 100
#     return score

# def calculate_emotion_consistency(facial_counts, total_faces, speech_counts, total_chunks, emotions):
#     if total_faces == 0 or total_chunks == 0:
#         return 0
#     facial_dist = {emo: facial_counts[emo] / total_faces for emo in emotions}
#     speech_dist = {emo: speech_counts[emo] / total_chunks for emo in emotions}
#     consistency = 1 - sum(abs(facial_dist[emo] - speech_dist[emo]) for emo in emotions) / 2
#     return consistency * 100

# def analyze_emotion_transitions(facial_timeline, speech_timeline, threshold=50.0):
#     transitions = {'facial': 0, 'speech': 0}
#     prev_emotion = {'facial': None, 'speech': None}

#     # Simplified: Check every 10th point or use all if < 100
#     facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
#     speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

#     # Facial transitions
#     for frame, emo, conf in facial_sampled:
#         if conf >= threshold and prev_emotion['facial'] and prev_emotion['facial'] != emo:
#             transitions['facial'] += 1
#         prev_emotion['facial'] = emo if conf >= threshold else prev_emotion['facial']

#     # Speech transitions
#     for time, emo, conf in speech_sampled:
#         if conf >= threshold and prev_emotion['speech'] and prev_emotion['speech'] != emo:
#             transitions['speech'] += 1
#         prev_emotion['speech'] = emo if conf >= threshold else prev_emotion['speech']

#     return transitions

# def analyze_peak_emotions(facial_timeline, speech_timeline, threshold=50.0):
#     peaks = {'facial': [], 'speech': []}
    
#     # Simplified: Sample every 10th point or use all if < 100
#     facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
#     speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

#     # Facial peaks (top 3 by confidence, simplified)
#     for frame, emo, conf in sorted(facial_sampled, key=lambda x: x[2], reverse=True):
#         if conf >= threshold and len(peaks['facial']) < 3:
#             peaks['facial'].append((frame, emo, conf))

#     # Speech peaks (top 3 by confidence, simplified)
#     for time, emo, conf in sorted(speech_sampled, key=lambda x: x[2], reverse=True):
#         if conf >= threshold and len(peaks['speech']) < 3:
#             peaks['speech'].append((time, emo, conf))

#     return peaks

# def analyze_emotion_duration(facial_timeline, speech_timeline, threshold=50.0):
#     durations = {'facial': Counter(), 'speech': Counter()}
#     current_emotion = {'facial': None, 'speech': None}
#     duration = {'facial': 0, 'speech': 0}

#     # Simplified: Sample every 10th point or use all if < 100
#     facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
#     speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

#     # Facial durations (simplified)
#     for frame, emo, conf in facial_sampled:
#         if conf >= threshold:
#             if current_emotion['facial'] == emo:
#                 duration['facial'] += 1
#             else:
#                 if current_emotion['facial']:
#                     durations['facial'][current_emotion['facial']] += duration['facial']
#                 current_emotion['facial'] = emo
#                 duration['facial'] = 1
#         else:
#             if current_emotion['facial']:
#                 durations['facial'][current_emotion['facial']] += duration['facial']
#                 current_emotion['facial'] = None
#                 duration['facial'] = 0
#     if current_emotion['facial']:
#         durations['facial'][current_emotion['facial']] += duration['facial']

#     # Speech durations (simplified)
#     for time, emo, conf in speech_sampled:
#         if conf >= threshold:
#             if current_emotion['speech'] == emo:
#                 duration['speech'] += 1
#             else:
#                 if current_emotion['speech']:
#                     durations['speech'][current_emotion['speech']] += duration['speech']
#                 current_emotion['speech'] = emo
#                 duration['speech'] = 1
#         else:
#             if current_emotion['speech']:
#                 durations['speech'][current_emotion['speech']] += duration['speech']
#                 current_emotion['speech'] = None
#                 duration['speech'] = 0
#     if current_emotion['speech']:
#         durations['speech'][current_emotion['speech']] += duration['speech']

#     return durations

# def generate_pdf_report(facial_overall, speech_overall, facial_feedback, speech_feedback, cumulative_feedback, plot_buf):
#     pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
#     try:
#         doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
#         styles = getSampleStyleSheet()
#         story = []

#         story.append(Paragraph("Interview Preparation Report", styles['Title']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Facial Emotion Distribution", styles['Heading2']))
#         for line in facial_overall:
#             story.append(Paragraph(line, styles['Normal']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Speech Emotion Distribution", styles['Heading2']))
#         for line in speech_overall:
#             story.append(Paragraph(line, styles['Normal']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Facial Feedback", styles['Heading2']))
#         story.append(Paragraph(facial_feedback, styles['Normal']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Speech Feedback", styles['Heading2']))
#         story.append(Paragraph(speech_feedback, styles['Normal']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Cumulative Feedback", styles['Heading2']))
#         story.append(Paragraph(cumulative_feedback, styles['Normal']))
#         story.append(Spacer(1, 12))
        
#         story.append(Paragraph("Emotion Trend Graph", styles['Heading2']))
#         story.append(Image(plot_buf, width=450, height=300))
        
#         doc.build(story)
#         return pdf_file.name
#     except Exception as e:
#         print(f"Error generating PDF: {str(e)}")
#         raise
#     finally:
#         try:
#             pdf_file.close()
#         except:
#             pass

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import tempfile
from collections import Counter

def plot_emotion_trend(facial_timeline, speech_timeline, emotions, threshold=50.0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Simple layout

    # Colors for emotions
    colors = {
        'Angry': 'red', 'Disgust': 'cyan', 'Fear': 'blue',
        'Happy': 'green', 'Sad': 'yellow', 'Surprise': 'orange', 'Neutral': 'purple'
    }

    # Facial emotions (simplified to max confidence per frame)
    facial_frames = [t[0] for t in facial_timeline]
    facial_emotions = [t[1] for t in facial_timeline]
    facial_confidences = [t[2] for t in facial_timeline]
    for emotion in emotions:
        y = [conf if emo == emotion and conf >= threshold else 0 for emo, conf in zip(facial_emotions, facial_confidences)]
        ax1.plot(facial_frames, y, label=f'Facial - {emotion}', color=colors[emotion], linewidth=1, alpha=0.7)
    ax1.set_title("Facial Emotion Trend (Confidence > 50%)", fontsize=12)
    ax1.set_xlabel("Time (frames)", fontsize=10)
    ax1.set_ylabel("Confidence (%)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Speech emotions (simplified to max confidence per chunk)
    speech_times = [t[0] for t in speech_timeline]
    speech_emotions = [t[1] for t in speech_timeline]
    speech_confidences = [t[2] for t in speech_timeline]
    for emotion in emotions:
        y = [conf if emo == emotion and conf >= threshold else 0 for emo, conf in zip(speech_emotions, speech_confidences)]
        ax2.plot(speech_times, y, label=f'Speech - {emotion}', color=colors[emotion], linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_title("Speech Emotion Trend (Confidence > 50%)", fontsize=12)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Confidence (%)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)  # Lower DPI for faster rendering
    buf.seek(0)
    plt.close()
    return buf

def plot_emotion_pie(counts, total, title):
    labels = [emo for emo in counts.keys() if counts[emo] > 0]
    sizes = [counts[emo] / total * 100 for emo in labels if counts[emo] > 0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    ax.axis('equal')
    ax.set_title(title, fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf

def calculate_readiness_score(facial_counts, total_faces, speech_counts, total_chunks):
    positive_emotions = ["Happy", "Neutral"]
    facial_positive = sum(facial_counts[emo] for emo in positive_emotions) / total_faces if total_faces > 0 else 0
    speech_positive = sum(speech_counts[emo] for emo in positive_emotions) / total_chunks if total_chunks > 0 else 0
    score = (facial_positive + speech_positive) / 2 * 100
    return score

def calculate_emotion_consistency(facial_counts, total_faces, speech_counts, total_chunks, emotions):
    if total_faces == 0 or total_chunks == 0:
        return 0
    facial_dist = {emo: facial_counts[emo] / total_faces for emo in emotions}
    speech_dist = {emo: speech_counts[emo] / total_chunks for emo in emotions}
    consistency = 1 - sum(abs(facial_dist[emo] - speech_dist[emo]) for emo in emotions) / 2
    return consistency * 100

def analyze_emotion_transitions(facial_timeline, speech_timeline, threshold=50.0):
    transitions = {'facial': 0, 'speech': 0}
    prev_emotion = {'facial': None, 'speech': None}

    # Simplified: Check every 10th point or use all if < 100
    facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
    speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

    # Facial transitions
    for frame, emo, conf in facial_sampled:
        if conf >= threshold and prev_emotion['facial'] and prev_emotion['facial'] != emo:
            transitions['facial'] += 1
        prev_emotion['facial'] = emo if conf >= threshold else prev_emotion['facial']

    # Speech transitions
    for time, emo, conf in speech_sampled:
        if conf >= threshold and prev_emotion['speech'] and prev_emotion['speech'] != emo:
            transitions['speech'] += 1
        prev_emotion['speech'] = emo if conf >= threshold else prev_emotion['speech']

    return transitions

def analyze_peak_emotions(facial_timeline, speech_timeline, threshold=50.0):
    peaks = {'facial': [], 'speech': []}
    
    # Simplified: Sample every 10th point or use all if < 100
    facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
    speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

    # Facial peaks (top 3 by confidence, simplified)
    for frame, emo, conf in sorted(facial_sampled, key=lambda x: x[2], reverse=True):
        if conf >= threshold and len(peaks['facial']) < 3:
            peaks['facial'].append((frame, emo, conf))

    # Speech peaks (top 3 by confidence, simplified)
    for time, emo, conf in sorted(speech_sampled, key=lambda x: x[2], reverse=True):
        if conf >= threshold and len(peaks['speech']) < 3:
            peaks['speech'].append((time, emo, conf))

    return peaks

def analyze_emotion_duration(facial_timeline, speech_timeline, threshold=50.0):
    durations = {'facial': Counter(), 'speech': Counter()}
    current_emotion = {'facial': None, 'speech': None}
    duration = {'facial': 0, 'speech': 0}

    # Simplified: Sample every 10th point or use all if < 100
    facial_sampled = facial_timeline[::10] if len(facial_timeline) > 100 else facial_timeline
    speech_sampled = speech_timeline[::10] if len(speech_timeline) > 100 else speech_timeline

    # Facial durations (simplified)
    for frame, emo, conf in facial_sampled:
        if conf >= threshold:
            if current_emotion['facial'] == emo:
                duration['facial'] += 1
            else:
                if current_emotion['facial']:
                    durations['facial'][current_emotion['facial']] += duration['facial']
                current_emotion['facial'] = emo
                duration['facial'] = 1
        else:
            if current_emotion['facial']:
                durations['facial'][current_emotion['facial']] += duration['facial']
                current_emotion['facial'] = None
                duration['facial'] = 0
    if current_emotion['facial']:
        durations['facial'][current_emotion['facial']] += duration['facial']

    # Speech durations (simplified)
    for time, emo, conf in speech_sampled:
        if conf >= threshold:
            if current_emotion['speech'] == emo:
                duration['speech'] += 1
            else:
                if current_emotion['speech']:
                    durations['speech'][current_emotion['speech']] += duration['speech']
                current_emotion['speech'] = emo
                duration['speech'] = 1
        else:
            if current_emotion['speech']:
                durations['speech'][current_emotion['speech']] += duration['speech']
                current_emotion['speech'] = None
                duration['speech'] = 0
    if current_emotion['speech']:
        durations['speech'][current_emotion['speech']] += duration['speech']

    return durations

def generate_pdf_report(facial_overall, speech_overall, facial_feedback, speech_feedback, cumulative_feedback, plot_buf, essay_question=None, essay_sentiment=None, essay_conf=None, essay_coherence=None, essay_feedback=None):
    """
    Generate a PDF report with facial, speech, essay, and cumulative feedback.
    
    Args:
        facial_overall: List of facial emotion percentages
        speech_overall: List of speech emotion percentages
        facial_feedback: Personalized facial feedback string
        speech_feedback: Personalized speech feedback string
        cumulative_feedback: Combined feedback string
        plot_buf: Buffer containing the emotion trend plot
        essay_question: The essay question (optional)
        essay_sentiment: Essay sentiment (optional)
        essay_conf: Essay sentiment confidence (optional)
        essay_coherence: Essay coherence score (optional)
        essay_feedback: Personalized essay feedback (optional)
    
    Returns:
        Path to the generated PDF file
    """
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        # Define custom styles for better formatting
        body_style = ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=6
        )
        story = []

        # Title
        story.append(Paragraph("Interview Preparation Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Facial Emotion Distribution
        story.append(Paragraph("Facial Emotion Distribution", styles['Heading2']))
        for line in facial_overall:
            story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 12))

        # Speech Emotion Distribution
        story.append(Paragraph("Speech Emotion Distribution", styles['Heading2']))
        for line in speech_overall:
            story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 12))

        # Facial Feedback
        story.append(Paragraph("Facial Feedback", styles['Heading2']))
        story.append(Paragraph(facial_feedback.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 12))

        # Speech Feedback
        story.append(Paragraph("Speech Feedback", styles['Heading2']))
        story.append(Paragraph(speech_feedback.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 12))

        # Essay Analysis (if provided)
        if essay_question and essay_sentiment is not None:
            story.append(Paragraph("Essay Analysis", styles['Heading2']))
            story.append(Paragraph(f"Question: {essay_question}", body_style))
            story.append(Paragraph(f"Sentiment: {essay_sentiment} ({essay_conf:.1f}% confidence)", body_style))
            story.append(Paragraph(f"Coherence Score: {essay_coherence:.1f}/100", body_style))
            story.append(Paragraph("Feedback:", body_style))
            story.append(Paragraph(essay_feedback.replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 12))

        # Cumulative Feedback
        story.append(Paragraph("Cumulative Feedback", styles['Heading2']))
        story.append(Paragraph(cumulative_feedback.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 12))

        # Emotion Trend Graph
        story.append(Paragraph("Emotion Trend Graph", styles['Heading2']))
        # Reset the buffer position to the beginning
        plot_buf.seek(0)
        story.append(Image(plot_buf, width=450, height=300))

        # Build the PDF
        doc.build(story)
        return pdf_file.name
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise
    finally:
        try:
            pdf_file.close()
        except:
            pass