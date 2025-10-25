from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
from backend.retrieval import ProductionChatbot
from backend.semantic_search import SemanticCodeSearch
import tempfile
import subprocess

load_dotenv()

app = FastAPI(title="CodeCompass")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbots = {}
repo_paths = {}

def clone_github_repo(github_url, target_dir = None):
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="repo_chatbot_")
    
    try:
        print(f"Cloning {github_url}...")
        subprocess.run(
            ['git', 'clone', '--depth', '1', github_url, target_dir],
            check=True,
            capture_output=True,
            timeout=60
        )

        print(f"Cloned to {target_dir}")
        return target_dir
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repo: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Clone timeout")

def get_repo_id_from_url(github_url):
    parts = github_url.rstrip('/').split('/')
    return f"{parts[-2]}_{parts[-1]}"

class FunctionDetail(BaseModel):
    name: str
    signature: Optional[str] = None
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None


class QueryRequest(BaseModel):
    repo_id: str
    repo_path: str
    user_query: str
    filtered_functions: List[FunctionDetail]


class QueryResponse(BaseModel):
    status: str
    response: str
    functions_count: int
    error: Optional[str] = None

semantic_search = SemanticCodeSearch()

@app.post("/initialise")
async def initialise_chatbot(repo_id, repo_path):
    try:
        chatbots[repo_id] = ProductionChatbot(repo_path)
        return {
            "status": "success",
            "message": f"Chatbot initialised for {repo_id}",
            "repo_path": repo_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        repo_id = get_repo_id_from_url(request.github_url)

        if repo_id not in repo_paths:
            print(f"Cloning new repo: {request.github_url}")
            repo_path = clone_github_repo(request.github_url)
            repo_paths[repo_id] = repo_path
            print(f"Cloned repo to: {repo_path}")
        else:
            print(f"Using cached repo: {repo_id}")
            repo_path = repo_paths[repo_id]
        
        if request.repo_id not in chatbots:
            chatbots[request.repo_id] = ProductionChatbot(request.repo_path)

        chatbot = chatbots[request.repo_id]

        filtered_functions = [func.model_dump() for func in request.filtered_functions]

        response = chatbot.generate_response(
            user_query=request.user_query,
            filtered_functions=filtered_functions
        )

        return QueryResponse(
            status="success",
            response=response,
            functions_count=len(filtered_functions)
        )
        
    except Exception as e:
        return QueryResponse(
            status="error",
            response="",
            functions_count=0,
            error=str(e)
        )
    
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_chatbots": len(chatbots)
    }

@app.delete("/cleanup/{repo_id}")
async def cleanup_chatbot(repo_id: str):
    if repo_id in chatbots:
        del chatbots[repo_id]
        return {"status": "success", "message": f"Cleaned up {repo_id}"}
    return {"status": "not_found"}

@app.post("/cleanup_all")
async def cleanup_all():
    """Clean up all chatbot instances"""
    count = len(chatbots)
    chatbots.clear()
    repo_paths.clear()
    
    return {
        "status": "success",
        "message": f"Cleaned up {count} chatbots"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)