import os
import sys
import json
import logging
import time
import nltk
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import preprocess_text, token_reduction_stats, count_tokens
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure all required NLTK data is downloaded
def ensure_nltk_data():
    """Download all required NLTK data packages."""
    logger.info("Ensuring all required NLTK data is downloaded...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab', quiet=True, raise_on_error=False)
    logger.info("NLTK data download complete")

# Call this function at the beginning to ensure all data is available
ensure_nltk_data()

# Sample texts of different types and lengths for testing
SAMPLE_TEXTS = {
    "legal_short": """
    This document is for informational purposes only. The parties hereby agree that all disputes arising out of this agreement shall be resolved through arbitration in accordance with the rules of the American Arbitration Association. This agreement is made and entered into on this day, January 1, 2023, by and between the undersigned parties.
    """,
    
    "legal_medium": """
    WHEREAS, the Parties desire to enter into this Agreement to define certain parameters of the future legal obligations, are bound by a duty of confidentiality with respect to their discussions and communications, and desire to specify the terms and conditions of their confidentiality obligations and to agree upon a procedure for the disclosure of Confidential Information. This document is confidential and proprietary. All rights reserved. This document may not be reproduced without the express written consent of all parties involved. The information contained herein is subject to change without notice and is not warranted to be error-free.
    
    NOW, THEREFORE, in consideration of the above premises, and for other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the Parties hereto agree as follows:
    
    1. Confidential Information. For purposes of this Agreement, "Confidential Information" shall include all information or material that has or could have commercial value or other utility in the business in which Disclosing Party is engaged.
    """,
    
    "technical_documentation": """
    The system architecture consists of three primary components: the frontend user interface, the backend API service, and the database layer. The frontend is built using React.js with Redux for state management. API requests are handled asynchronously using Axios. The backend is implemented in Python using the FastAPI framework, which provides automatic OpenAPI documentation. Database interactions are managed through SQLAlchemy ORM with PostgreSQL as the primary data store. Authentication is implemented using JWT tokens with refresh token rotation for enhanced security.
    """,
    
    "conversational": """
    Hi there! I'm looking for some information about your product. I've been comparing different options and I'm particularly interested in the pricing structure and whether there are any discounts for annual subscriptions. Also, could you tell me a bit more about the customer support options? Is there 24/7 support available or is it limited to business hours? Thanks in advance for your help!
    """
}

def test_nltk_preprocessing_effectiveness() -> Dict:
    """
    Test the effectiveness of NLTK preprocessing by comparing token counts
    before and after preprocessing for different types of text.
    
    Returns:
        Dict: Results of the test including token reduction statistics
    """
    logger.info("Starting NLTK preprocessing effectiveness test")
    
    results = {
        "overall": {
            "original_tokens": 0,
            "processed_tokens": 0,
            "tokens_reduced": 0,
            "reduction_percent": 0,
            "processing_time": 0
        },
        "samples": {}
    }
    
    total_start_time = time.time()
    
    # Process each sample text
    for text_type, text in SAMPLE_TEXTS.items():
        logger.info(f"Processing {text_type} sample...")
        
        # Measure processing time
        start_time = time.time()
        processed_text = preprocess_text(text)
        processing_time = time.time() - start_time
        
        # Calculate token reduction statistics
        stats = token_reduction_stats(text, processed_text)
        
        # Store results for this sample
        results["samples"][text_type] = {
            **stats,
            "processing_time": processing_time,
            "original_text": text,
            "processed_text": processed_text
        }
        
        # Update overall statistics
        results["overall"]["original_tokens"] += stats["original_tokens"]
        results["overall"]["processed_tokens"] += stats["processed_tokens"]
        results["overall"]["tokens_reduced"] += stats["tokens_reduced"]
    
    # Calculate overall reduction percentage
    if results["overall"]["original_tokens"] > 0:
        results["overall"]["reduction_percent"] = (
            results["overall"]["tokens_reduced"] / results["overall"]["original_tokens"]
        ) * 100
    
    results["overall"]["processing_time"] = time.time() - total_start_time
    
    return results

def test_rag_token_usage(api_key: str = None) -> Dict:
    """
    Test the token usage in a RAG system with and without NLTK preprocessing.
    This simulates how the preprocessing affects the actual token usage when
    sending to the LLM.
    
    Args:
        api_key: OpenAI API key (optional, will use from .env if not provided)
        
    Returns:
        Dict: Results of the token usage test
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please provide it as an argument or set it in .env file.")
    
    client = OpenAI(api_key=api_key)
    
    logger.info("Starting RAG token usage test")
    
    results = {
        "overall": {
            "without_nltk": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost": 0
            },
            "with_nltk": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost": 0
            },
            "token_savings": 0,
            "cost_savings": 0,
            "savings_percent": 0
        },
        "samples": {}
    }
    
    # Process each sample text
    for text_type, text in SAMPLE_TEXTS.items():
        logger.info(f"Testing RAG token usage for {text_type} sample...")
        
        # Simulate RAG context with original text
        query = "What is the main topic of this document?"
        context_without_nltk = f"Context: {text}\n\nQuery: {query}"
        
        # Process text with NLTK
        processed_text = preprocess_text(text)
        context_with_nltk = f"Context: {processed_text}\n\nQuery: {query}"
        
        # Get token usage from OpenAI API
        try:
            # Without NLTK
            response_without_nltk = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": context_without_nltk}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # With NLTK
            response_with_nltk = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": context_with_nltk}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # Extract token usage
            usage_without_nltk = {
                "total_tokens": response_without_nltk.usage.total_tokens,
                "prompt_tokens": response_without_nltk.usage.prompt_tokens,
                "completion_tokens": response_without_nltk.usage.completion_tokens,
                "estimated_cost": calculate_cost(response_without_nltk.usage)
            }
            
            usage_with_nltk = {
                "total_tokens": response_with_nltk.usage.total_tokens,
                "prompt_tokens": response_with_nltk.usage.prompt_tokens,
                "completion_tokens": response_with_nltk.usage.completion_tokens,
                "estimated_cost": calculate_cost(response_with_nltk.usage)
            }
            
            # Calculate savings
            token_savings = usage_without_nltk["total_tokens"] - usage_with_nltk["total_tokens"]
            cost_savings = usage_without_nltk["estimated_cost"] - usage_with_nltk["estimated_cost"]
            savings_percent = (token_savings / usage_without_nltk["total_tokens"]) * 100 if usage_without_nltk["total_tokens"] > 0 else 0
            
            # Store results for this sample
            results["samples"][text_type] = {
                "without_nltk": usage_without_nltk,
                "with_nltk": usage_with_nltk,
                "token_savings": token_savings,
                "cost_savings": cost_savings,
                "savings_percent": savings_percent,
                "response_without_nltk": response_without_nltk.choices[0].message.content,
                "response_with_nltk": response_with_nltk.choices[0].message.content
            }
            
            # Update overall statistics
            for key in ["total_tokens", "prompt_tokens", "completion_tokens", "estimated_cost"]:
                results["overall"]["without_nltk"][key] += usage_without_nltk[key]
                results["overall"]["with_nltk"][key] += usage_with_nltk[key]
            
            results["overall"]["token_savings"] += token_savings
            results["overall"]["cost_savings"] += cost_savings
            
        except Exception as e:
            logger.error(f"Error testing RAG token usage for {text_type}: {str(e)}")
            results["samples"][text_type] = {"error": str(e)}
    
    # Calculate overall savings percentage
    if results["overall"]["without_nltk"]["total_tokens"] > 0:
        results["overall"]["savings_percent"] = (
            results["overall"]["token_savings"] / results["overall"]["without_nltk"]["total_tokens"]
        ) * 100
    
    return results

def calculate_cost(usage) -> float:
    """
    Calculate the estimated cost of API usage based on current OpenAI pricing.
    
    Args:
        usage: OpenAI API usage object
        
    Returns:
        float: Estimated cost in USD
    """
    # Current pricing for gpt-3.5-turbo as of my knowledge cutoff
    prompt_price_per_1k = 0.0015  # $0.0015 per 1K tokens
    completion_price_per_1k = 0.002  # $0.002 per 1K tokens
    
    prompt_cost = (usage.prompt_tokens / 1000) * prompt_price_per_1k
    completion_cost = (usage.completion_tokens / 1000) * completion_price_per_1k
    
    return prompt_cost + completion_cost

def run_tests_and_save_results() -> None:
    """
    Run all tests and save the results to a JSON file.
    """
    results = {
        "nltk_preprocessing": test_nltk_preprocessing_effectiveness()
    }
    
    try:
        results["rag_token_usage"] = test_rag_token_usage()
    except Exception as e:
        logger.error(f"Error running RAG token usage test: {str(e)}")
        results["rag_token_usage"] = {"error": str(e)}
    
    # Save results to file
    output_file = Path("nltk_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("NLTK INTEGRATION TEST RESULTS SUMMARY")
    print("=" * 50)
    
    # NLTK preprocessing summary
    nltk_results = results["nltk_preprocessing"]["overall"]
    print(f"\nNLTK Preprocessing Effectiveness:")
    print(f"  Original tokens: {nltk_results['original_tokens']}")
    print(f"  Processed tokens: {nltk_results['processed_tokens']}")
    print(f"  Tokens reduced: {nltk_results['tokens_reduced']}")
    print(f"  Reduction percentage: {nltk_results['reduction_percent']:.2f}%")
    print(f"  Processing time: {nltk_results['processing_time']:.4f} seconds")
    
    # RAG token usage summary if available
    if "error" not in results.get("rag_token_usage", {}):
        rag_results = results["rag_token_usage"]["overall"]
        print(f"\nRAG Token Usage Comparison:")
        print(f"  Without NLTK: {rag_results['without_nltk']['total_tokens']} tokens (${rag_results['without_nltk']['estimated_cost']:.6f})")
        print(f"  With NLTK: {rag_results['with_nltk']['total_tokens']} tokens (${rag_results['with_nltk']['estimated_cost']:.6f})")
        print(f"  Token savings: {rag_results['token_savings']} tokens")
        print(f"  Cost savings: ${rag_results['cost_savings']:.6f}")
        print(f"  Savings percentage: {rag_results['savings_percent']:.2f}%")
    else:
        print(f"\nRAG Token Usage Test Error: {results['rag_token_usage']['error']}")
    
    print("\nDetailed results saved to nltk_test_results.json")

if __name__ == "__main__":
    run_tests_and_save_results()