import re
import string
import nltk
import logging
from typing import List, Set
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data packages if not already downloaded."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK data already downloaded")
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        logger.info("NLTK data downloaded successfully")

# Common legal boilerplate phrases to filter out
BOILERPLATE_PHRASES = [
    "this document is for informational purposes only",
    "this agreement is made and entered into",
    "in witness whereof",
    "for all intents and purposes",
    "please read this document carefully",
    "this document contains confidential information",
    "all rights reserved",
    "subject to change without notice",
    "this document supersedes all previous",
    "for internal use only",
    "confidential and proprietary",
    "do not distribute",
    "for discussion purposes only",
    "this is not legal advice",
    "this document does not constitute",
    "this document is not intended to",
    "this document may not be reproduced",
    "this document is the property of",
    "this document is subject to",
    "this document is protected by",
    "this document is confidential",
    "this document is provided as is",
    "this document is not a contract",
    "this document is not a commitment",
    "this document is not a guarantee",
    "this document is not a warranty",
    "this document is not a representation",
    "this document is not an offer",
    "this document is not a promise",
    "this document is not a solicitation",
    "this document is not an advertisement",
    "this document is not a recommendation",
    "this document is not an endorsement",
    "this document is not a certification",
    "this document is not a license",
    "this document is not a permit",
    "this document is not a registration",
    "this document is not a qualification",
    "this document is not a consent",
    "this document is not a waiver",
    "this document is not a release",
    "this document is not a discharge",
    "this document is not a covenant",
    "this document is not a condition",
    "this document is not a restriction",
    "this document is not a limitation",
    "this document is not a prohibition",
    "this document is not a reservation",
    "this document is not a disclaimer"
]

def remove_stopwords(text: str) -> str:
    """Remove common English stopwords from text.
    
    Args:
        text: Input text to process
        
    Returns:
        Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def lemmatize_text(text: str) -> str:
    """Apply lemmatization to normalize words.
    
    Args:
        text: Input text to process
        
    Returns:
        Text with words lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_text)

def clean_punctuation(text: str) -> str:
    """Remove punctuation and special characters while preserving sentence boundaries.
    
    Args:
        text: Input text to process
        
    Returns:
        Text with punctuation cleaned
    """
    # Split text into sentences to preserve sentence boundaries
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    
    for sentence in sentences:
        # Remove punctuation except periods
        cleaned = re.sub(f'[{re.escape(string.punctuation.replace(".", ""))}]', ' ', sentence)
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_sentences.append(cleaned)
    
    # Join sentences with periods
    return '. '.join(cleaned_sentences)

def remove_boilerplate(text: str) -> str:
    """Remove common legal boilerplate sentences.
    
    Args:
        text: Input text to process
        
    Returns:
        Text with boilerplate sentences removed
    """
    sentences = sent_tokenize(text)
    filtered_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        is_boilerplate = False
        
        # Check if sentence contains any boilerplate phrase
        for phrase in BOILERPLATE_PHRASES:
            if phrase in sentence_lower:
                is_boilerplate = True
                break
        
        if not is_boilerplate:
            filtered_sentences.append(sentence)
    
    return ' '.join(filtered_sentences)

def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to reduce token count while preserving meaning.
    
    Args:
        text: Input text to process
        
    Returns:
        Preprocessed text with reduced token count
    """
    # Ensure NLTK data is downloaded
    download_nltk_data()
    
    # Apply preprocessing steps in sequence
    logger.info("Removing boilerplate text...")
    text = remove_boilerplate(text)
    
    logger.info("Cleaning punctuation...")
    text = clean_punctuation(text)
    
    logger.info("Removing stopwords...")
    text = remove_stopwords(text)
    
    logger.info("Applying lemmatization...")
    text = lemmatize_text(text)
    
    return text

def count_tokens(text: str) -> int:
    """Count the approximate number of tokens in text.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Simple approximation: split by whitespace and count
    return len(text.split())

def token_reduction_stats(original_text: str, processed_text: str) -> dict:
    """Calculate token reduction statistics.
    
    Args:
        original_text: Original text before preprocessing
        processed_text: Text after preprocessing
        
    Returns:
        Dictionary with token reduction statistics
    """
    original_count = count_tokens(original_text)
    processed_count = count_tokens(processed_text)
    reduction = original_count - processed_count
    reduction_percent = (reduction / original_count) * 100 if original_count > 0 else 0
    
    return {
        "original_tokens": original_count,
        "processed_tokens": processed_count,
        "tokens_reduced": reduction,
        "reduction_percent": reduction_percent
    }