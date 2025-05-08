#essay_processing.py

import re
import time

# Simple word lists for sentiment analysis (rule-based)
POSITIVE_WORDS = {"good", "great", "excellent", "positive", "strong", "success", "achieve", "benefit", "clear", "well", "improve", "value", "nurture", "build"}
NEGATIVE_WORDS = {"bad", "poor", "weak", "fail", "problem", "issue", "misunderstand", "difficult", "challenge", "negative"}

# List of common transition words for coherence heuristic
TRANSITION_WORDS = {
    "however", "therefore", "moreover", "furthermore", "consequently", "thus", "nevertheless",
    "similarly", "in addition", "on the other hand", "for example", "in contrast", "as a result"
}

def calculate_coherence(essay_text):
    """
    Calculate a coherence score based on sentence length variation and transition word usage.
    Returns a score between 0 and 100.
    """
    print("Calculating coherence...")
    try:
        # Split the essay into sentences
        sentences = re.split(r'[.!?]+', essay_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            print("No sentences found in essay.")
            return 0.0
        
        # Calculate sentence length variation
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) <= 1:
            length_variation_score = 0
        else:
            length_variation = max(sentence_lengths) - min(sentence_lengths)
            length_variation_score = min(length_variation * 5, 50)  # Scale variation to 0-50
        
        # Count transition words
        essay_lower = essay_text.lower()
        transition_count = sum(1 for word in TRANSITION_WORDS if word in essay_lower)
        transition_score = min(transition_count * 10, 50)  # Scale transition usage to 0-50
        
        # Combine scores
        coherence_score = length_variation_score + transition_score
        print(f"Coherence score: {coherence_score}")
        return coherence_score
    except Exception as e:
        print(f"Error in calculate_coherence: {str(e)}")
        raise

def process_essay(essay_text):
    """
    Process the essay text to get sentiment, confidence, and coherence scores using a simple rule-based approach.
    Returns: (sentiment, confidence, coherence)
    """
    print("Starting essay processing...")
    start_time = time.time()
    
    # Simple rule-based sentiment analysis
    try:
        words = essay_text.lower().split()
        positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment = "Neutral"
            confidence = 50.0
        else:
            positive_ratio = positive_count / total_sentiment_words
            if positive_ratio > 0.6:
                sentiment = "Positive"
                confidence = positive_ratio * 100
            elif positive_ratio < 0.4:
                sentiment = "Negative"
                confidence = (1 - positive_ratio) * 100
            else:
                sentiment = "Neutral"
                confidence = 50.0
        print("Sentiment analysis completed")
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        raise
    
    # Calculate coherence
    try:
        coherence = calculate_coherence(essay_text)
        print("Coherence calculation completed")
    except Exception as e:
        print(f"Error during coherence calculation: {str(e)}")
        raise
    
    print(f"Essay processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}%, Coherence: {coherence:.2f}")
    return sentiment, confidence, coherence

def get_essay_feedback(essay_text):
    """
    Generate feedback based on sentiment and coherence scores.
    """
    print("Generating essay feedback...")
    try:
        sentiment, confidence, coherence = process_essay(essay_text)
        
        feedback = f"Sentiment: {sentiment}\n"
        feedback += f"Confidence: {confidence:.2f}%\n"
        feedback += f"Coherence Score: {coherence:.2f}/100\n"
        
        if coherence < 50:
            feedback += "The essay could benefit from improved clarity and structure. Consider using more transition words and varying sentence lengths."
        elif coherence < 75:
            feedback += "The essay has decent coherence but could be improved. Try adding more transition words to connect ideas smoothly."
        else:
            feedback += "The essay is well-structured and coherent. Great job!"
        
        print("Essay feedback generated")
        return feedback
    except Exception as e:
        print(f"Error in get_essay_feedback: {str(e)}")
        raise

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