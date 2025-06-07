from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai

app = FastAPI()
load_dotenv(dotenv_path=".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - path relative to backend directory
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("../frontend/index.html")

# Add API health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "devcascade-api"}

class Command(BaseModel):
    command: str

def format_response(text):
    """Format the response to ensure it's in a single paragraph"""
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
        
        full_prompt = system_prompt + cmd.command
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        
        # Format the response to ensure single paragraph structure
        formatted_response = format_response(response.text)
        
        return {"response": formatted_response}
        
    except Exception as e:
        return {"error": str(e)}
