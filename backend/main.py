from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai

app = FastAPI()

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if frontend directory exists and mount static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    async def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="Frontend not found")
else:
    @app.get("/")
    async def root():
        return {"message": "DevCascade API is running!", "status": "healthy", "note": "Frontend not available"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "devcascade-api"}

class Command(BaseModel):
    command: str

def format_response(text):
    """Format the response to ensure it's in a single paragraph"""
    if not text:
        return "Sorry, I couldn't generate a response."
    # Remove excessive line breaks and format as single paragraph
    formatted = ' '.join(text.strip().split())
    return formatted

def check_name_query(command):
    """Check if the user is asking about the bot's name"""
    name_keywords = ['name', 'who are you', 'what are you called', 'your name']
    command_lower = command.lower()
    return any(keyword in command_lower for keyword in name_keywords)

@app.post("/api/command")
async def handle_command(cmd: Command):
    try:
        # Validate input
        if not cmd.command or not cmd.command.strip():
            raise HTTPException(status_code=400, detail="Command cannot be empty")
        
        # Check if user is asking about the bot's name
        if check_name_query(cmd.command):
            return {"response": "My name is devcascade. I'm here to help you with your questions and tasks."}
        
        system_prompt = """
        You are a helpful AI assistant. Please follow these guidelines:
        1. Always respond in a single, well-structured paragraph and also Use Bold letters to Highlight the Important Things.
        2. Be concise but informative
        3. Avoid using bullet points or multiple paragraphs
        4. Keep your response clear and coherent
        5. If the response would naturally be long, summarize the key points in paragraph form
        6. Be polite everytime and Dont use any thing rude 
        7. Ask the user everytime after giving the output whether he wants anything else.
        
        User query: """
        
        full_prompt = system_prompt + cmd.command.strip()
        
        # Initialize model and generate response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        
        # Check if response was generated successfully
        if not response or not response.text:
            return {"error": "Failed to generate response from AI model"}
        
        # Format the response to ensure single paragraph structure
        formatted_response = format_response(response.text)
        
        return {"response": formatted_response}
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging (in production, use proper logging)
        print(f"Error in handle_command: {str(e)}")
        return {"error": "An internal error occurred. Please try again."}

# For Render deployment - this is important
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
