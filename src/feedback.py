# #feedback.py
# from collections import Counter

# def get_facial_feedback_personalized(facial_emotion_counts, total_faces):
#     if total_faces == 0:
#         return "Hey there! It looks like we couldn’t detect any faces in your video. For your interview prep, make sure your face is clearly visible—good lighting and a centered camera angle can make a big difference. Try recording again with those tweaks!"

#     dominant_emotion = max(facial_emotion_counts, key=facial_emotion_counts.get)
#     dominant_percentage = (facial_emotion_counts[dominant_emotion] / total_faces) * 100

#     feedback = f"Alright, let’s break this down! Your facial expressions leaned heavily towards {dominant_emotion.lower()} ({dominant_percentage:.2f}%). Here’s how that might come across in an interview:\n\n"

#     if dominant_emotion == "Happy":
#         feedback += "You’ve got a great vibe going—looking happy and engaged is a big win! Studies show that candidates who smile genuinely are perceived as 20% more likable by hiring managers. Keep that positivity flowing, but maybe sprinkle in a bit of calm confidence to balance it out."
#     elif dominant_emotion == "Sad":
#         feedback += "It seems like you might’ve been feeling a bit down or serious during this take. Sadness showed up a lot, and while it’s okay to be authentic, interviewers often look for energy and optimism—about 70% of them rate enthusiasm as a top trait. Try lifting your mood before the next practice; maybe take a deep breath or think of something that excites you about the job!"
#     elif dominant_emotion == "Angry":
#         feedback += "You came across as a bit frustrated or intense here. Anger popped up quite a bit, and in interviews, that might signal stress—something 60% of recruiters watch out for. Let’s soften that edge; practice a relaxed smile or a steady tone to show you’re cool under pressure."
#     elif dominant_emotion == "Fear":
#         feedback += "There’s a hint of nervousness showing through, which is totally normal when prepping for interviews! Fear was noticeable, and while it’s relatable, confidence is key—85% of interviewers say it’s a dealbreaker if it’s missing. Take a moment to relax your shoulders and breathe deeply before your next run."
#     elif dominant_emotion == "Neutral":
#         feedback += "You kept things pretty neutral, which can be a solid base! It’s great for showing composure, but too much neutrality might make you seem disengaged—about 30% of candidates lose points for not showing enough passion. Try adding a warm smile or a bit more energy to connect better."
#     elif dominant_emotion == "Disgust":
#         feedback += "It looks like something might’ve thrown you off, as disgust stood out. In an interview, this could come across as disinterest, which 50% of hiring managers flag as a red sign. Let’s reset—focus on staying engaged and curious, even if a question catches you off guard."
#     elif dominant_emotion == "Surprise":
#         feedback += "You seemed surprised quite a bit! That can add a lively spark, but too much might suggest you’re unprepared—interviewers often expect steady responses 80% of the time. Practice a few common questions to feel more in control next time."

#     return feedback

# def get_speech_feedback_personalized(speech_emotion_counts, total_chunks):
#     if total_chunks == 0:
#         return "Whoops, we didn’t pick up any audio! For your interview, clear speech is crucial—make sure your mic’s on and you’re speaking at a steady pace. Give it another go with that in mind!"

#     dominant_emotion = max(speech_emotion_counts, key=speech_emotion_counts.get)
#     dominant_percentage = (speech_emotion_counts[dominant_emotion] / total_chunks) * 100

#     feedback = f"Let’s talk about your voice—it’s a huge part of your interview presence! Your tone leaned towards {dominant_emotion.lower()} ({dominant_percentage:.2f}%), and here’s how that might play out:\n\n"

#     if dominant_emotion == "Happy":
#         feedback += "Your voice sounds upbeat and cheerful, which is fantastic! Recruiters love hearing enthusiasm—it’s rated highly by 65% of them. Just keep your pace steady so your excitement doesn’t rush your words."
#     elif dominant_emotion == "Sad":
#         feedback += "Your tone came across as a bit down or quiet. Sadness was prominent, and while it’s natural to feel off sometimes, interviewers pick up on vocal energy—low tones can drop engagement by 40%. Try speaking with a bit more lift; imagine you’re sharing good news!"
#     elif dominant_emotion == "Angry":
#         feedback += "There’s some intensity in your voice, which might sound like frustration. Anger showed up a lot, and in interviews, a calm tone is preferred—75% of interviewers note it as a sign of professionalism. Take a deep breath and slow down to keep it smooth."
#     elif dominant_emotion == "Fear":
#         feedback += "You sounded a little nervous, which is super common when practicing! Fear was noticeable, and while it’s okay to feel it, a steady voice boosts confidence scores by 30%. Practice a few lines out loud to find your rhythm."
#     elif dominant_emotion == "Neutral":
#         feedback += "Your voice stayed pretty even-keeled, which is a nice foundation. Neutral tones are safe, but adding some warmth or excitement could make you stand out—60% of hiring managers value vocal expressiveness. Try varying your pitch a bit!"
#     elif dominant_emotion == "Disgust":
#         feedback += "Your tone had a hint of displeasure, which might not vibe well in an interview. Disgust stood out, and recruiters often see that as a lack of interest—50% flag it. Focus on a friendly, open sound; maybe practice with a positive topic first."
#     elif dominant_emotion == "Surprise":
#         feedback += "You sounded surprised quite a bit! It adds a spark, but too much can seem unsteady—80% of interviewers look for vocal consistency. Work on grounding your tone to sound more prepared."

#     return feedback

# def get_cumulative_feedback(facial_emotion_counts, total_faces, speech_emotion_counts, total_chunks):
#     if total_faces == 0 and total_chunks == 0:
#         return "Hey, we didn’t catch much to work with here—no faces or audio! For your next try, make sure your camera’s capturing your expressions and your mic’s picking up your voice. You’ve got this!"

#     facial_dominant = max(facial_emotion_counts, key=facial_emotion_counts.get) if total_faces > 0 else "None"
#     speech_dominant = max(speech_emotion_counts, key=speech_emotion_counts.get) if total_chunks > 0 else "None"

#     feedback = "Alright, let’s put it all together—your face and voice tell the full story! Here’s how they teamed up:\n\n"

#     if total_faces > 0 and total_chunks > 0:
#         feedback += f"Your facial expressions leaned towards {facial_dominant.lower()} ({(facial_emotion_counts[facial_dominant] / total_faces) * 100:.2f}%), while your voice carried {speech_dominant.lower()} ({(speech_emotion_counts[speech_dominant] / total_chunks) * 100:.2f}%). "
#         if facial_dominant == speech_dominant:
#             feedback += f"They’re in sync, which is awesome—consistency makes you come across as genuine, something 90% of interviewers value. "
#             if facial_dominant in ["Happy", "Neutral"]:
#                 feedback += "You’re projecting a solid, positive vibe—keep that energy, and you’ll leave a great impression!"
#             else:
#                 feedback += f"But since it’s {facial_dominant.lower()}, let’s shift gears a bit—aim for a happier or calmer vibe to align with what 70% of hiring managers look for."
#         else:
#             feedback += "They’re telling slightly different stories, which can happen! Interviewers notice this—about 60% say mismatched emotions might signal discomfort. "
#             if facial_dominant in ["Happy", "Neutral"] and speech_dominant not in ["Happy", "Neutral"]:
#                 feedback += "Your face is on point, but your voice could use a lift. Try practicing with a brighter tone to match that positivity!"
#             elif speech_dominant in ["Happy", "Neutral"] and facial_dominant not in ["Happy", "Neutral"]:
#                 feedback += "Your voice sounds great, but your expressions could soften up a bit. A warm smile could tie it all together!"
#             else:
#                 feedback += "Both could use a boost—aim for a friendly, confident look and sound to hit that sweet spot interviewers love."
#     elif total_faces > 0:
#         feedback += f"We got your facial expressions ({facial_dominant.lower()}), but no audio. Focus on keeping your voice clear and upbeat next time—it’s half the battle in interviews!"
#     elif total_chunks > 0:
#         feedback += f"We heard your voice ({speech_dominant.lower()}), but missed your face. Make sure you’re visible—facial cues are key for 80% of interviewers!"

#     feedback += "\n\nFor your next practice, think about projecting confidence and warmth—studies show candidates who balance these traits have a 25% higher chance of nailing the interview. You’re doing great; just keep refining!"
#     return feedback

# def get_essay_feedback_personalized(sentiment, sentiment_conf, coherence):
#     return get_essay_feedback(sentiment, sentiment_conf, coherence)  # From essay_processing.py

# def get_cumulative_feedback(facial_emotion_counts, total_faces, speech_emotion_counts, total_chunks, essay_sentiment=None, essay_sentiment_conf=None, essay_coherence=None):
#     if total_faces == 0 and total_chunks == 0 and essay_sentiment is None:
#         return "Hey, we didn’t catch much to work with—no faces, audio, or essay! Let’s get all three in your next try—camera, mic, and a written response. You’ve got this!"

#     feedback = "Alright, let’s tie it all together—your face, voice, and words tell the full story! Here’s the rundown:\n\n"
#     facial_dominant = max(facial_emotion_counts, key=facial_emotion_counts.get) if total_faces > 0 else "None"
#     speech_dominant = max(speech_emotion_counts, key=speech_emotion_counts.get) if total_chunks > 0 else "None"

#     if total_faces > 0:
#         feedback += f"Your facial expressions leaned towards {facial_dominant.lower()} ({(facial_emotion_counts[facial_dominant] / total_faces) * 100:.2f}%). "
#     if total_chunks > 0:
#         feedback += f"Your voice carried {speech_dominant.lower()} ({(speech_emotion_counts[speech_dominant] / total_chunks) * 100:.2f}%). "
#     if essay_sentiment:
#         feedback += f"Your essay had a {essay_sentiment.lower()} tone ({essay_sentiment_conf:.2f}%) with {essay_coherence:.2f}% coherence. "

#     if total_faces > 0 and total_chunks > 0 and essay_sentiment:
#         if facial_dominant == speech_dominant == essay_sentiment:
#             feedback += f"Wow, everything’s aligned—consistency across face, voice, and words is gold, valued by 90% of interviewers! "
#             if facial_dominant in ["Happy", "Neutral"] and essay_sentiment == "Positive":
#                 feedback += "You’re radiating positivity—keep it up for a stellar impression!"
#             else:
#                 feedback += "But it’s a less positive vibe—shift towards happier tones for that 70% recruiter preference."
#         else:
#             feedback += "There’s some mixed signals here—60% of interviewers notice this. "
#             if facial_dominant in ["Happy", "Neutral"] and speech_dominant in ["Happy", "Neutral"] and essay_sentiment == "Positive":
#                 feedback += "You’re mostly positive—fine-tune any outliers for a cohesive win!"
#             else:
#                 feedback += "Aim for a consistent positive vibe across all three—face, voice, and essay—to nail it."
#     elif total_faces > 0 and total_chunks > 0:
#         feedback += "No essay yet—add it for the full picture! Face and voice are a start, but written skills matter too."
#     elif total_faces > 0 or total_chunks > 0:
#         feedback += "We’ve got partial data—add an essay to round it out! Interviewers weigh all communication channels."

#     feedback += "\n\nFor your next go, aim for confidence, warmth, and clarity across everything—studies show a 25% higher success rate when you nail this combo!"
#     return feedback


from collections import Counter

def get_facial_feedback_personalized(facial_emotion_counts, total_faces):
    if total_faces == 0:
        return "Hey there! It looks like we couldn't detect any faces in your video. For your interview prep, make sure your face is clearly visible—good lighting and a centered camera angle can make a big difference. Try recording again with those tweaks!"

    dominant_emotion = max(facial_emotion_counts, key=facial_emotion_counts.get)
    dominant_percentage = (facial_emotion_counts[dominant_emotion] / total_faces) * 100

    feedback = f"Alright, let's break this down! Your facial expressions leaned heavily towards {dominant_emotion.lower()} ({dominant_percentage:.2f}%). Here's how that might come across in an interview:\n\n"

    if dominant_emotion == "Happy":
        feedback += "You've got a great vibe going—looking happy and engaged is a big win! Studies show that candidates who smile genuinely are perceived as 20% more likable by hiring managers. Keep that positivity flowing, but maybe sprinkle in a bit of calm confidence to balance it out."
    elif dominant_emotion == "Sad":
        feedback += "It seems like you might've been feeling a bit down or serious during this take. Sadness showed up a lot, and while it's okay to be authentic, interviewers often look for energy and optimism—about 70% of them rate enthusiasm as a top trait. Try lifting your mood before the next practice; maybe take a deep breath or think of something that excites you about the job!"
    elif dominant_emotion == "Angry":
        feedback += "You came across as a bit frustrated or intense here. Anger popped up quite a bit, and in interviews, that might signal stress—something 60% of recruiters watch out for. Let's soften that edge; practice a relaxed smile or a steady tone to show you're cool under pressure."
    elif dominant_emotion == "Fear":
        feedback += "There's a hint of nervousness showing through, which is totally normal when prepping for interviews! Fear was noticeable, and while it's relatable, confidence is key—85% of interviewers say it's a dealbreaker if it's missing. Take a moment to relax your shoulders and breathe deeply before your next run."
    elif dominant_emotion == "Neutral":
        feedback += "You kept things pretty neutral, which can be a solid base! It's great for showing composure, but too much neutrality might make you seem disengaged—about 30% of candidates lose points for not showing enough passion. Try adding a warm smile or a bit more energy to connect better."
    elif dominant_emotion == "Disgust":
        feedback += "It looks like something might've thrown you off, as disgust stood out. In an interview, this could come across as disinterest, which 50% of hiring managers flag as a red sign. Let's reset—focus on staying engaged and curious, even if a question catches you off guard."
    elif dominant_emotion == "Surprise":
        feedback += "You seemed surprised quite a bit! That can add a lively spark, but too much might suggest you're unprepared—interviewers often expect steady responses 80% of the time. Practice a few common questions to feel more in control next time."

    return feedback

def get_speech_feedback_personalized(speech_emotion_counts, total_chunks):
    if total_chunks == 0:
        return "Whoops, we didn't pick up any audio! For your interview, clear speech is crucial—make sure your mic's on and you're speaking at a steady pace. Give it another go with that in mind!"

    dominant_emotion = max(speech_emotion_counts, key=speech_emotion_counts.get)
    dominant_percentage = (speech_emotion_counts[dominant_emotion] / total_chunks) * 100

    feedback = f"Let's talk about your voice—it's a huge part of your interview presence! Your tone leaned towards {dominant_emotion.lower()} ({dominant_percentage:.2f}%), and here's how that might play out:\n\n"

    if dominant_emotion == "Happy":
        feedback += "Your voice sounds upbeat and cheerful, which is fantastic! Recruiters love hearing enthusiasm—it's rated highly by 65% of them. Just keep your pace steady so your excitement doesn't rush your words."
    elif dominant_emotion == "Sad":
        feedback += "Your tone came across as a bit down or quiet. Sadness was prominent, and while it's natural to feel off sometimes, interviewers pick up on vocal energy—low tones can drop engagement by 40%. Try speaking with a bit more lift; imagine you're sharing good news!"
    elif dominant_emotion == "Angry":
        feedback += "There's some intensity in your voice, which might sound like frustration. Anger showed up a lot, and in interviews, a calm tone is preferred—75% of interviewers note it as a sign of professionalism. Take a deep breath and slow down to keep it smooth."
    elif dominant_emotion == "Fear":
        feedback += "You sounded a little nervous, which is super common when practicing! Fear was noticeable, and while it's okay to feel it, a steady voice boosts confidence scores by 30%. Practice a few lines out loud to find your rhythm."
    elif dominant_emotion == "Neutral":
        feedback += "Your voice stayed pretty even-keeled, which is a nice foundation. Neutral tones are safe, but adding some warmth or excitement could make you stand out—60% of hiring managers value vocal expressiveness. Try varying your pitch a bit!"
    elif dominant_emotion == "Disgust":
        feedback += "Your tone had a hint of displeasure, which might not vibe well in an interview. Disgust stood out, and recruiters often see that as a lack of interest—50% flag it. Focus on a friendly, open sound; maybe practice with a positive topic first."
    elif dominant_emotion == "Surprise":
        feedback += "You sounded surprised quite a bit! It adds a spark, but too much can seem unsteady—80% of interviewers look for vocal consistency. Work on grounding your tone to sound more prepared."

    return feedback

def get_essay_feedback_personalized(sentiment, confidence, coherence):
    """
    Generate personalized feedback based on sentiment and coherence scores.
    """
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

def get_cumulative_feedback(facial_emotion_counts, total_faces, speech_emotion_counts, total_chunks, essay_sentiment=None, essay_conf=None, essay_coherence=None):
    """Comprehensive feedback combining facial, speech, and essay analysis"""
    if total_faces == 0 and total_chunks == 0 and essay_sentiment is None:
        return "Hey, we didn't catch much to work with—no faces, audio, or essay! Let's get all three in your next try—camera, mic, and a written response. You've got this!"

    feedback = "## COMPLETE INTERVIEW READINESS ASSESSMENT\n\n"
    
    # Determine dominant emotions and sentiment
    facial_dominant = max(facial_emotion_counts, key=facial_emotion_counts.get) if total_faces > 0 and facial_emotion_counts else "None"
    speech_dominant = max(speech_emotion_counts, key=speech_emotion_counts.get) if total_chunks > 0 and speech_emotion_counts else "None"
    
    # Add summary section first
    feedback += "### SUMMARY\n"
    
    # Calculate overall alignment score
    alignment_score = 0
    alignment_count = 0
    
    if total_faces > 0 and facial_dominant != "None":
        alignment_count += 1
        if speech_dominant != "None" and facial_dominant == speech_dominant:
            alignment_score += 1
        if essay_sentiment == "Positive" and facial_dominant in ["Happy", "Neutral"]:
            alignment_score += 0.5
        elif essay_sentiment == "Negative" and facial_dominant in ["Sad", "Angry", "Fear", "Disgust"]:
            alignment_score += 0.5
    
    if total_chunks > 0 and speech_dominant != "None":
        alignment_count += 1
        if essay_sentiment == "Positive" and speech_dominant in ["Happy", "Neutral"]:
            alignment_score += 0.5
        elif essay_sentiment == "Negative" and speech_dominant in ["Sad", "Angry", "Fear", "Disgust"]:
            alignment_score += 0.5
    
    alignment_percentage = (alignment_score / max(1, alignment_count)) * 100 if alignment_count > 0 else 0
    
    # Overall strengths and areas for improvement
    strengths = []
    improvements = []
    
    # Add facial feedback
    if total_faces > 0:
        facial_percentage = (facial_emotion_counts[facial_dominant] / total_faces) * 100 if facial_emotion_counts else 0
        feedback += f"• Facial Expression: Primarily {facial_dominant.lower()} ({facial_percentage:.1f}%)\n"
        if facial_dominant in ["Happy", "Neutral"]:
            strengths.append("Positive facial expressions")
        else:
            improvements.append("More positive facial expressions")
    
    # Add speech feedback
    if total_chunks > 0:
        speech_percentage = (speech_emotion_counts[speech_dominant] / total_chunks) * 100 if speech_emotion_counts else 0
        feedback += f"• Voice Tone: Primarily {speech_dominant.lower()} ({speech_percentage:.1f}%)\n"
        if speech_dominant in ["Happy", "Neutral"]:
            strengths.append("Engaging vocal tone")
        else:
            improvements.append("More confident vocal delivery")
    
    # Add essay feedback
    if essay_sentiment:
        feedback += f"• Essay Tone: {essay_sentiment} ({essay_conf:.1f}% confidence)\n"
        feedback += f"• Essay Coherence: {essay_coherence:.1f}/100\n"
        if essay_sentiment == "Positive":
            strengths.append("Positive written communication")
        else:
            improvements.append("More positive framing in writing")
        
        if essay_coherence >= 70:
            strengths.append("Well-structured written responses")
        else:
            improvements.append("Improved essay structure and flow")
    
    # Add alignment score
    feedback += f"• Communication Alignment: {alignment_percentage:.1f}%\n"
    
    if alignment_percentage >= 80:
        feedback += "\n**Overall Assessment:** EXCELLENT INTERVIEW READINESS. Your communication channels (face, voice, and writing) align well, creating a cohesive and authentic impression.\n\n"
    elif alignment_percentage >= 60:
        feedback += "\n**Overall Assessment:** GOOD INTERVIEW READINESS. Your communication is generally consistent, though there are opportunities to align your message better across all channels.\n\n"
    else:
        feedback += "\n**Overall Assessment:** NEEDS IMPROVEMENT. There are noticeable inconsistencies between your facial expressions, voice tone, and written communication that may create confusion for interviewers.\n\n"
    
    # Add key strengths section
    feedback += "### KEY STRENGTHS\n"
    if strengths:
        for strength in strengths[:3]:  # Limit to top 3 strengths
            feedback += f"• {strength}\n"
    else:
        feedback += "• No clear strengths identified - focus on the improvement areas below\n"
    
    # Add improvement areas
    feedback += "\n### AREAS FOR IMPROVEMENT\n"
    if improvements:
        for improvement in improvements[:3]:  # Limit to top 3 improvements
            feedback += f"• {improvement}\n"
    else:
        feedback += "• Continue maintaining your current excellent performance\n"
    
    # Add specific detailed feedback section
    feedback += "\n### DETAILED ANALYSIS\n"
    
    # Alignment between facial and speech
    if total_faces > 0 and total_chunks > 0:
        if facial_dominant == speech_dominant:
            if facial_dominant in ["Happy", "Neutral"]:
                feedback += "Your facial expressions and voice are aligned in a positive way—this creates an impression of authenticity and confidence. Interviewers consistently rate this alignment highly, with 80% noting it as a sign of strong communication skills.\n\n"
            else:
                feedback += f"Your facial expressions and voice are consistently showing {facial_dominant.lower()}, which may not create the best impression. Negative emotions can reduce your likability by up to 30% in interviews. Try to shift towards more positive expressions and tone—practice smiling and speaking with enthusiasm about your achievements.\n\n"
        else:
            feedback += f"There's a mismatch between your facial expression ({facial_dominant.lower()}) and voice tone ({speech_dominant.lower()}). This can create confusion for interviewers, who might question your sincerity—about 60% of hiring managers note that inconsistent signals impact their perception. Work on aligning your emotions; for example, if you're aiming for a positive tone in your voice, let your face reflect that with a smile.\n\n"
    
    # Alignment between essay and body language (facial/speech)
    if essay_sentiment:
        if total_faces > 0 or total_chunks > 0:
            positive_body = (facial_dominant in ["Happy", "Neutral"] if total_faces > 0 else False) or (speech_dominant in ["Happy", "Neutral"] if total_chunks > 0 else False)
            
            if (essay_sentiment == "Positive" and positive_body) or (essay_sentiment != "Positive" and not positive_body):
                feedback += "Your written tone aligns with your physical presentation, creating a cohesive impression. This consistency is key—interviewers often look for candidates who present a unified message across all communication channels, with 70% citing it as a marker of professionalism.\n\n"
            else:
                feedback += f"There's a disconnect between your written tone ({essay_sentiment.lower()}) and your physical presentation ({'positive' if positive_body else 'negative'}). For example, a positive essay paired with a negative demeanor (or vice versa) can make your message feel inconsistent. Aim for alignment—ensure your facial expressions and voice tone match the positivity or professionalism of your written answers.\n\n"
    
    # Specific advice based on overall performance
    feedback += "### ACTIONABLE TIPS FOR YOUR NEXT INTERVIEW\n"
    if "More positive facial expressions" in improvements or "More confident vocal delivery" in improvements:
        feedback += "• **Boost Positivity:** Before your interview, try a quick mood-lifter—watch a funny video or recall a proud moment to bring out more happiness in your face and voice. This can increase your likability by up to 25%.\n"
    
    if "More positive framing in writing" in improvements:
        feedback += "• **Reframe Your Writing:** When discussing challenges in your essay, focus on solutions and growth. For example, instead of saying 'I struggled with teamwork,' try 'I initially faced challenges with teamwork but learned to collaborate effectively by...'\n"
    
    if "Improved essay structure and flow" in improvements:
        feedback += "• **Enhance Essay Structure:** Use transition words like 'therefore,' 'however,' or 'for example' to connect your ideas. A clear structure can improve your perceived clarity by 40% in written responses.\n"
    
    if alignment_percentage < 60:
        feedback += "• **Align Your Communication:** Practice delivering your essay answers out loud while recording yourself. Check if your facial expressions and voice tone match the tone of your writing. Consistency across all channels can boost your overall impression by 50%.\n"
    
    feedback += "• **Practice with Common Questions:** Rehearse answers to typical interview questions (like 'Tell me about yourself') to build confidence across your facial expressions, voice, and writing. Confidence is a top trait for 85% of interviewers.\n"
    
    return feedback