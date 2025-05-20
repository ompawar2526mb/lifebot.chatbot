import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMHandler:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM handler.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"LLM handler initialized with model: {model_name}")
    
    def generate_response(self, 
                         query: str, 
                         context: List[Dict], 
                         dual_response: bool = False,
                         conversation_history: Optional[str] = None) -> Dict:
        """
        Generate a response using the LLM.
        
        Args:
            query: User query
            context: Retrieved context from RAG system
            dual_response: Whether to generate dual responses
            conversation_history: Optional conversation history string
            
        Returns:
            Dict: Response containing text and optional audio
        """
        try:
            # Format the context
            context_text = "\n\n".join([f"Document {i+1}:\n{chunk['text']}" 
                                      for i, chunk in enumerate(context)])
            
            # Create the system message with context and conversation history
            system_message = "You are a helpful AI assistant. Use the following context to answer the user's question."
            if conversation_history:
                system_message += f"\n\n{conversation_history}"
            if context_text:
                system_message += f"\n\nRelevant context:\n{context_text}"
            
            if dual_response:
                # Generate dual responses
                system_message += "\n\nPlease provide two different responses:\n1. A concise response\n2. A more detailed, structured response"
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                response_text = response.choices[0].message.content.strip()
                responses = []
                
                if "Concise Response:" in response_text and "Structured Response:" in response_text:
                    parts = response_text.split("Structured Response:")
                    responses.append(parts[0].replace("Concise Response:", "").strip())
                    responses.append(parts[1].strip())
                else:
                    responses = response_text.split("\n\n", 1)
                    if len(responses) < 2:
                        responses.append("I apologize, I couldn't generate a second response.")
                
                return {"responses": responses, "audio": None}
            else:
                # Single response
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return {"responses": [response.choices[0].message.content.strip()], "audio": None}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"responses": ["I apologize, but I encountered an error while generating the response. Please try again."], "audio": None}