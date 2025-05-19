import os
from openai import OpenAI
from dotenv import load_dotenv
from pdf_processor import get_vector_store, query_pdf

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_input: str, history: list, pdf_context: str = "", dual_response: bool = False) -> dict:
    """
    Generate a response using OpenAI's GPT model, incorporating PDF context if available.
    Supports dual responses with different tones if dual_response is True.
    
    Args:
        user_input (str): The user's input message
        history (list): List of previous messages in the conversation
        pdf_context (str): Relevant context from PDF query (optional)
        dual_response (bool): Whether to generate two responses with different tones
    
    Returns:
        dict: Contains 'responses' (list of responses) and 'audio' (None in this case)
    """
    try:
        # Query the PDF for relevant information
        pdf_response = query_pdf(user_input)
        
        # Define a single comprehensive system prompt
        system_prompt = """You are a helpful AI assistant that provides answers based on the provided document context. 
        Always prioritize using the document context to answer questions. If the document context is relevant, use it as your primary source.
        
        When using document context:
        1. Quote relevant parts when appropriate
        2. Explain the context clearly
        3. Connect the context to the user's question
        
        When the document context isn't relevant:
        1. Acknowledge that the information isn't in the document
        2. Provide a helpful general response
        3. Suggest what kind of information might be more relevant
        
        IF YOU ARE ASKED FOR A SPECIFIC LANGUAGE, PROVIDE THE OUTPUT IN THAT LANGUAGE AND ITS FONT.
        ALSO MAKE HINDI, BANGLA, AND GUJRATI LANGUAGES AVAILABLE.
        """
        
        # Append PDF context if available
        if pdf_response:
            system_prompt += f"\n\nHere is the relevant information from the document:\n{pdf_response}"
        
        # Prepare messages for the API call
        messages = []
        
        # Add conversation history
        for msg in history:
            role = msg.get("role", "user")
            if role == "bot":
                role = "assistant"
            messages.append({
                "role": role,
                "content": msg.get("text", "")
            })
        
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        if dual_response:
            # Request two responses with different styles using the same system prompt
            system_message = (
                f"{system_prompt}\n\n"
                "Provide two distinct responses to the user's input:\n"
                "1. A concise, straightforward response (2-3 sentences, casual tone)\n"
                "2. A structured, formal response with clear explanations and organized points\n"
                "Label them as 'Concise Response:' and 'Structured Response:'\n"
                "If applicable, include formatting such as **bold**, *italic*, - bullet points, or LaTeX for formulas (e.g., $$E = mc^2$$)."
            )
            
            messages.insert(0, {"role": "system", "content": system_message})
            
            # Generate response using OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1500,  # Increased token limit for two responses
                temperature=0.7
            )
            
            # Parse the response to extract the two parts
            response_text = response.choices[0].message.content.strip()
            responses = ["", ""]
            
            # Split the response based on labels
            if "Concise Response:" in response_text and "Structured Response:" in response_text:
                parts = response_text.split("Structured Response:")
                responses[0] = parts[0].replace("Concise Response:", "").strip()
                responses[1] = parts[1].strip()
            else:
                # Fallback: if the response doesn't follow the expected format, split evenly
                responses = response_text.split("\n\n", 1)
                if len(responses) < 2:
                    responses.append("I apologize, I couldn't generate a second response.")
            
            return {"responses": responses, "audio": None}
        else:
            # Use the system prompt for single response
            messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Generate response using OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return {"responses": [response.choices[0].message.content.strip()], "audio": None}
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return {"responses": ["I apologize, but I encountered an error while generating a response. Please try again."], "audio": None}