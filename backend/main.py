from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DevCascade API", version="1.0.0")

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY environment variable is required")
    raise ValueError("GEMINI_API_KEY environment variable is required")

try:
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# CORS middleware - More restrictive for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Static files and frontend serving
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    logger.info(f"Serving static files from: {frontend_dir}")
    
    @app.get("/")
    async def serve_index():
        """Serve the main frontend application"""
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            logger.warning("Frontend index.html not found")
            raise HTTPException(status_code=404, detail="Frontend not found")
else:
    logger.warning("Frontend directory not found")
    @app.get("/")
    async def root():
        return {
            "message": "DevCascade API is running!", 
            "status": "healthy", 
            "note": "Frontend not available",
            "endpoints": {
                "health": "/health",
                "api": "/api/command",
                "docs": "/docs"
            }
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test Gemini API connection
        model = genai.GenerativeModel("gemini-1.5-flash")
        test_response = model.generate_content("Hello")
        
        return {
            "status": "healthy", 
            "service": "devcascade-api",
            "gemini_api": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "service": "devcascade-api",
            "error": str(e),
            "version": "1.0.0"
        }

class Command(BaseModel):
    command: str
    
    class Config:
        schema_extra = {
            "example": {
                "command": "What is DevCascade?"
            }
        }

def format_response(text: str) -> str:
    """Format the response to ensure it's in a single paragraph"""
    if not text:
        return "Sorry, I couldn't generate a response."
    
    # Remove excessive line breaks and format as single paragraph
    formatted = ' '.join(text.strip().split())
    return formatted

def check_name_query(command: str) -> bool:
    """Check if the user is asking about the bot's name"""
    name_keywords = ['name', 'who are you', 'what are you called', 'your name', 'devcascade']
    command_lower = command.lower()
    return any(keyword in command_lower for keyword in name_keywords)

@app.post("/api/command")
async def handle_command(cmd: Command):
    """Handle user commands and generate AI responses"""
    try:
        # Validate input
        if not cmd.command or not cmd.command.strip():
            raise HTTPException(status_code=400, detail="Command cannot be empty")
        
        command_text = cmd.command.strip()
        logger.info(f"Processing command: {command_text[:50]}...")
        
        # Check if user is asking about the bot's name
        if check_name_query(command_text):
            return {
                "response": "My name is **DevCascade**. I'm here to help you with your questions and tasks. Is there anything else you'd like to know?"
            }
        
        # Enhanced system prompt
        system_prompt = """
        You are DevCascade, a helpful AI assistant. Please follow these guidelines:
        1. Always respond in a single, well-structured paragraph
        2. Use **bold** formatting to highlight important things
        3. Be concise but informative
        4. Avoid using bullet points or multiple paragraphs
        5. Keep your response clear and coherent
        6. If the response would naturally be long, summarize the key points in paragraph form
        7. Be polite and professional at all times
        8. End your response by asking if the user needs anything else
        9. If asked about technical topics, provide practical and actionable advice
        10. Stay focused on being helpful and solution-oriented
        
        User query: """
        
        full_prompt = system_prompt + command_text
        
        # Initialize model and generate response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        
        # Check if response was generated successfully
        if not response or not response.text:
            logger.error("No response generated from Gemini API")
            return {"error": "Failed to generate response from AI model"}
        
        # Format the response to ensure single paragraph structure
        formatted_response = format_response(response.text)
        
        logger.info("Response generated successfully")
        return {"response": formatted_response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in handle_command: {str(e)}")
        return {
            "error": "An internal error occurred. Please try again.",
            "details": str(e) if os.getenv("DEBUG") == "true" else None
        }

@app.get("/api/status")
async def get_status():
    """Get current API status"""
    return {
        "status": "online",
        "service": "devcascade-api",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "command": "/api/command",
            "status": "/api/status"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/api/command", "/docs"]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"Starting DevCascade API on {host}:{port}")
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )
