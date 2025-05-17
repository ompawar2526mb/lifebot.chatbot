def validate_speech_input(text):
    """
    Validate speech input to ensure it's not empty or invalid.
    
    Args:
        text (str): The recognized speech text.
        
    Returns:
        str: Validated text, or None if invalid.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return None
    return text.strip()