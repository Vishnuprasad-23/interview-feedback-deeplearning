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

# Test the essay processing
if __name__ == "__main__":
    essay_text = "In professional settings, strong communication skills lead to better teamwork, productivity, and leadership. Employers value individuals who can express their ideas clearly and listen to others constructively. Similarly, in personal relationships, good communication nurtures strong bonds and prevents misunderstandings. In conclusion, mastering communication skills is crucial for personal and professional growth. By being clear, empathetic, and an active listener, one can build stronger relationships and achieve greater success in life."
    try:
        sentiment, confidence, coherence = process_essay(essay_text)
        print(f"Final Result - Sentiment: {sentiment}, Confidence: {confidence:.2f}%, Coherence: {coherence:.2f}")
    except Exception as e:
        print(f"Error in test script: {str(e)}")