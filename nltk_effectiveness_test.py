import os
import sys
import json
import logging
import time
import nltk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tabulate import tabulate

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
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download punkt_tab: {e}")
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
    """,
    
    "academic": """
    The study examined the effects of climate change on biodiversity in tropical rainforests. Results indicated a significant decline in species richness over the 10-year observation period. Statistical analysis revealed a strong negative correlation between rising temperatures and population densities of endemic species. Furthermore, the research demonstrated that habitat fragmentation exacerbated these effects, with isolated forest patches showing accelerated biodiversity loss compared to contiguous forest areas. These findings suggest that conservation efforts should prioritize maintaining forest connectivity to mitigate the impacts of climate change on tropical ecosystems.
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

def test_semantic_preservation() -> Dict:
    """
    Test whether the semantic meaning is preserved after NLTK preprocessing.
    This uses a simple similarity check between responses to the same query
    with and without preprocessing.
    
    Returns:
        Dict: Results of the semantic preservation test
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set it in .env file.")
    
    client = OpenAI(api_key=api_key)
    
    logger.info("Starting semantic preservation test")
    
    results = {
        "overall": {
            "semantic_preservation_score": 0,
            "samples_tested": 0
        },
        "samples": {}
    }
    
    # Process each sample text
    for text_type, text in SAMPLE_TEXTS.items():
        logger.info(f"Testing semantic preservation for {text_type} sample...")
        
        # Process text with NLTK
        processed_text = preprocess_text(text)
        
        # Define a set of test questions
        test_questions = [
            "What is the main topic of this document?",
            "Summarize the key points in this text.",
            "What is the purpose of this document?",
            "What are the key findings of this research?",
            "What are the implications of this study for the field of ecology?"
            "What are the main conclusions of this study?",
            "What are the implications of this study for the field of astrophysics?",
            "What are the main conclusions of this research on biodiversity loss?",
            "What are the implications of this analysis for the development of machine learning algorithms?",
            "What are the key findings of this study on sustainable agriculture practices?",
            "What are the implications of this research for the field of cognitive neuroscience?",
            "What are the main outcomes of this investigation into solar energy efficiency?",
            "What are the implications of this study for marine ecosystem conservation?",
            "What are the primary conclusions of this research on blockchain technology?",
            "What are the implications of this analysis for the field of public health policy?",
            "What are the key conclusions of this study on exoplanet habitability?",
            
        ]
        
        sample_results = []
        
        for question in test_questions:
            try:
                # Get response with original text
                response_original = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}"}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent responses
                    max_tokens=150
                )
                
                # Get response with processed text
                response_processed = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {processed_text}\n\nQuestion: {question}"}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent responses
                    max_tokens=150
                )
                
                original_answer = response_original.choices[0].message.content.strip()
                processed_answer = response_processed.choices[0].message.content.strip()
                
                # Compare the two answers using a similarity check
                # Here we use a simple approach - ask GPT to rate the similarity
                similarity_check = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates text similarity."},
                        {"role": "user", "content": f"On a scale of 0 to 10, where 0 means completely different and 10 means semantically identical, rate the similarity between these two answers to the same question:\n\nQuestion: {question}\n\nAnswer 1: {original_answer}\n\nAnswer 2: {processed_answer}\n\nProvide only a number as your response."}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                
                similarity_score = similarity_check.choices[0].message.content.strip()
                # Extract just the number from the response
                similarity_score = float(''.join(c for c in similarity_score if c.isdigit() or c == '.') or 0) / 10
                
                sample_results.append({
                    "question": question,
                    "original_answer": original_answer,
                    "processed_answer": processed_answer,
                    "similarity_score": similarity_score
                })
                
                # Update overall statistics
                results["overall"]["semantic_preservation_score"] += similarity_score
                results["overall"]["samples_tested"] += 1
                
            except Exception as e:
                logger.error(f"Error testing semantic preservation for {text_type}, question '{question}': {str(e)}")
                sample_results.append({
                    "question": question,
                    "error": str(e)
                })
        
        results["samples"][text_type] = sample_results
    
    # Calculate overall semantic preservation score
    if results["overall"]["samples_tested"] > 0:
        results["overall"]["semantic_preservation_score"] /= results["overall"]["samples_tested"]
    
    return results

def generate_visualizations(results: Dict) -> None:
    """
    Generate visualizations of the test results.
    
    Args:
        results: Test results dictionary
    """
    # Create output directory for visualizations
    output_dir = Path("nltk_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Token reduction by text type
    if "nltk_preprocessing" in results:
        plt.figure(figsize=(12, 6))
        
        samples = results["nltk_preprocessing"]["samples"]
        text_types = list(samples.keys())
        original_tokens = [samples[t]["original_tokens"] for t in text_types]
        processed_tokens = [samples[t]["processed_tokens"] for t in text_types]
        
        x = np.arange(len(text_types))
        width = 0.35
        
        plt.bar(x - width/2, original_tokens, width, label='Original')
        plt.bar(x + width/2, processed_tokens, width, label='After NLTK')
        
        plt.xlabel('Text Type')
        plt.ylabel('Token Count')
        plt.title('Token Reduction by Text Type')
        plt.xticks(x, [t.replace('_', ' ').title() for t in text_types])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, (orig, proc) in enumerate(zip(original_tokens, processed_tokens)):
            reduction = ((orig - proc) / orig) * 100 if orig > 0 else 0
            plt.text(i, max(orig, proc) + 5, f"{reduction:.1f}% reduction", 
                     ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "token_reduction_by_text_type.png")
        plt.close()
    
    # 2. Cost savings visualization
    if "rag_token_usage" in results:
        plt.figure(figsize=(10, 6))
        
        samples = results["rag_token_usage"]["samples"]
        text_types = [t for t in samples.keys() if "error" not in samples[t]]
        
        if text_types:  # Only proceed if we have valid samples
            without_nltk_cost = [samples[t]["without_nltk"]["estimated_cost"] for t in text_types]
            with_nltk_cost = [samples[t]["with_nltk"]["estimated_cost"] for t in text_types]
            
            x = np.arange(len(text_types))
            width = 0.35
            
            plt.bar(x - width/2, without_nltk_cost, width, label='Without NLTK')
            plt.bar(x + width/2, with_nltk_cost, width, label='With NLTK')
            
            plt.xlabel('Text Type')
            plt.ylabel('Estimated Cost (USD)')
            plt.title('Cost Comparison by Text Type')
            plt.xticks(x, [t.replace('_', ' ').title() for t in text_types])
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for i, (orig, proc) in enumerate(zip(without_nltk_cost, with_nltk_cost)):
                savings = ((orig - proc) / orig) * 100 if orig > 0 else 0
                plt.text(i, max(orig, proc) + 0.0001, f"{savings:.1f}% savings", 
                         ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "cost_savings_by_text_type.png")
            plt.close()
    
    # 3. Semantic preservation visualization
    if "semantic_preservation" in results:
        plt.figure(figsize=(10, 6))
        
        samples = results["semantic_preservation"]["samples"]
        text_types = list(samples.keys())
        
        avg_scores = []
        for text_type in text_types:
            sample_results = samples[text_type]
            valid_scores = [r["similarity_score"] for r in sample_results if "similarity_score" in r]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            avg_scores.append(avg_score)
        
        plt.bar(range(len(text_types)), avg_scores, color='skyblue')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
        
        plt.xlabel('Text Type')
        plt.ylabel('Semantic Preservation Score (0-1)')
        plt.title('Semantic Preservation by Text Type')
        plt.xticks(range(len(text_types)), [t.replace('_', ' ').title() for t in text_types])
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, score in enumerate(avg_scores):
            plt.text(i, score + 0.02, f"{score:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "semantic_preservation_by_text_type.png")
        plt.close()

def print_tabular_results(results: Dict) -> None:
    """
    Print the test results in a tabular format.
    
    Args:
        results: Test results dictionary
    """
    print("\n" + "=" * 80)
    print("NLTK INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    # 1. Token reduction results
    if "nltk_preprocessing" in results:
        print("\n1. TOKEN REDUCTION RESULTS\n")
        
        # Overall results
        overall = results["nltk_preprocessing"]["overall"]
        print(f"Overall Token Reduction: {overall['reduction_percent']:.2f}%")
        print(f"Original Tokens: {overall['original_tokens']}")
        print(f"Processed Tokens: {overall['processed_tokens']}")
        print(f"Tokens Reduced: {overall['tokens_reduced']}")
        print(f"Processing Time: {overall['processing_time']:.4f} seconds\n")
        
        # Results by text type
        samples = results["nltk_preprocessing"]["samples"]
        table_data = []
        headers = ["Text Type", "Original Tokens", "Processed Tokens", "Tokens Reduced", "Reduction %", "Time (s)"]
        
        for text_type, stats in samples.items():
            table_data.append([
                text_type.replace('_', ' ').title(),
                stats["original_tokens"],
                stats["processed_tokens"],
                stats["tokens_reduced"],
                f"{stats['reduction_percent']:.2f}%",
                f"{stats['processing_time']:.4f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 2. RAG token usage results
    if "rag_token_usage" in results:
        print("\n2. RAG TOKEN USAGE RESULTS\n")
        
        # Overall results
        overall = results["rag_token_usage"]["overall"]
        print(f"Overall Token Savings: {overall['savings_percent']:.2f}%")
        print(f"Without NLTK: {overall['without_nltk']['total_tokens']} tokens (${overall['without_nltk']['estimated_cost']:.6f})")
        print(f"With NLTK: {overall['with_nltk']['total_tokens']} tokens (${overall['with_nltk']['estimated_cost']:.6f})")
        print(f"Tokens Saved: {overall['token_savings']}")
        print(f"Cost Savings: ${overall['cost_savings']:.6f}\n")
        
        # Results by text type
        samples = results["rag_token_usage"]["samples"]
        table_data = []
        headers = ["Text Type", "Without NLTK", "With NLTK", "Tokens Saved", "Cost Savings", "Savings %"]
        
        for text_type, stats in samples.items():
            if "error" in stats:
                continue
                
            table_data.append([
                text_type.replace('_', ' ').title(),
                f"{stats['without_nltk']['total_tokens']} (${stats['without_nltk']['estimated_cost']:.6f})",
                f"{stats['with_nltk']['total_tokens']} (${stats['with_nltk']['estimated_cost']:.6f})",
                stats["token_savings"],
                f"${stats['cost_savings']:.6f}",
                f"{stats['savings_percent']:.2f}%"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 3. Semantic preservation results
    if "semantic_preservation" in results:
        print("\n3. SEMANTIC PRESERVATION RESULTS\n")
        
        # Overall results
        overall = results["semantic_preservation"]["overall"]
        print(f"Overall Semantic Preservation Score: {overall['semantic_preservation_score']:.2f} (0-1 scale)")
        print(f"Samples Tested: {overall['samples_tested']}\n")
        
        # Results by text type
        samples = results["semantic_preservation"]["samples"]
        for text_type, sample_results in samples.items():
            print(f"\n{text_type.replace('_', ' ').title()}:")
            
            table_data = []
            headers = ["Question", "Similarity Score"]
            
            for result in sample_results:
                if "error" in result:
                    continue
                    
                table_data.append([
                    result["question"],
                    f"{result['similarity_score']:.2f}"
                ])
            
            print(tabulate(table_data, headers=headers, tablefmt="simple"))

def run_tests_and_save_results(run_semantic_test: bool = False) -> None:
    """
    Run all tests and save the results to a JSON file.
    
    Args:
        run_semantic_test: Whether to run the semantic preservation test
    """
    results = {
        "nltk_preprocessing": test_nltk_preprocessing_effectiveness()
    }
    
    try:
        results["rag_token_usage"] = test_rag_token_usage()
    except Exception as e:
        logger.error(f"Error running RAG token usage test: {str(e)}")
        results["rag_token_usage"] = {"error": str(e)}
    
    if run_semantic_test:
        try:
            results["semantic_preservation"] = test_semantic_preservation()
        except Exception as e:
            logger.error(f"Error running semantic preservation test: {str(e)}")
            results["semantic_preservation"] = {"error": str(e)}
    
    # Save results to file
    output_dir = Path("nltk_test_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "nltk_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_file}")
    
    # Generate visualizations
    try:
        generate_visualizations(results)
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    # Print tabular results
    print_tabular_results(results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the effectiveness of NLTK integration in the RAG system")
    parser.add_argument("--semantic", action="store_true", help="Run semantic preservation test (uses more API calls)")
    args = parser.parse_args()
    
    run_tests_and_save_results(run_semantic_test=args.semantic)