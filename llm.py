import os
from openai import OpenAI
from dotenv import load_dotenv
from pdf_processor import get_vector_store, query_pdf

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_input: str, history: list, pdf_context: str = "") -> str:
    """
    Generate a response using OpenAI's GPT model, incorporating PDF context if available.
    
    Args:
        user_input (str): The user's input message
        history (list): List of previous messages in the conversation
        pdf_context (str): Relevant context from PDF query (optional)
    
    Returns:
        str: The generated response
    """
    try:
        # Query the PDF for relevant information
        pdf_response = query_pdf(user_input)
        
        # Prepare system message with PDF context
        system_message = """You are a helpful AI assistant that specializes in answering questions based on the provided document context. 
        Always prioritize using the document context to answer questions. If the document context is relevant, use it as your primary source.
        If the document context doesn't contain the answer, you can provide a general response, but make it clear that you're not using the document information.
        
        When using document context:
        1. Quote relevant parts when appropriate
        2. Explain the context in your own words
        3. Connect the context to the user's question
        
        When the document context isn't relevant:
        1. Acknowledge that the information isn't in the document
        2. Provide a helpful general response
        3. Suggest what kind of information might be more relevant

 
        if query contains :
        response1 :qwertyuiop
        resopnse2 :asdfghjkl
        """
        
        if pdf_response:
            system_message += f"\n\nHere is the relevant information from the document:\n{pdf_response}"
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add conversation history
        for msg in history:
            # Ensure role is one of: system, user, assistant
            role = msg.get("role", "user")
            if role == "bot":
                role = "assistant"
            messages.append({
                "role": role,
                "content": msg.get("text", "")
            })
        
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate response using OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,  # Increased token limit for more detailed responses
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again."