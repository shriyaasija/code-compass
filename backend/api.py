from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
import tempfile
import subprocess
from pathlib import Path

from backend.retrieval import ProductionChatbot
from backend.pageindex_semantic_search import PageIndexSemanticSearch


app = FastAPI(title="CodeCompass - PageIndex Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
chatbots = {}  # repo_id -> ProductionChatbot
repo_paths = {}  # repo_id -> local path
semantic_search = PageIndexSemanticSearch(embedding_model="all-MiniLM-L6-v2")

def clone_github_repo(github_url: str, target_dir: str = None) -> str:
    """Clone a GitHub repository locally."""
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="repo_chatbot_")

    try:
        print(f"📥 Cloning {github_url}...")
        subprocess.run(
            ['git', 'clone', '--depth', '1', github_url, target_dir],
            check=True,
            capture_output=True,
            timeout=60
        )
        print(f"✅ Cloned to {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repo: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Clone timeout (60s exceeded)")


def get_repo_id_from_url(github_url: str) -> str:
    """Extract repo_id from GitHub URL."""
    parts = github_url.rstrip('/').split('/')
    return f"{parts[-2]}_{parts[-1]}"

class InitializeRequest(BaseModel):
    """Request to initialize a repository."""
    github_url: Optional[str] = None
    repo_path: Optional[str] = None
    json_tree_path: str  # Path to PageIndex JSON file
    repo_id: Optional[str] = None  # If not provided, will be auto-generated


class QueryRequest(BaseModel):
    """Simplified query request - just repo_id and query!"""
    repo_id: str
    user_query: str
    top_k: Optional[int] = 5  # Number of functions to retrieve


class QueryResponse(BaseModel):
    """Response with LLM-generated answer."""
    status: str
    response: str
    matched_functions: List[dict]  # Functions that were matched
    functions_count: int
    error: Optional[str] = None

@app.post("/initialize")
async def initialize_repository(request: InitializeRequest):
    """
    Initialize a repository for querying.

    Steps:
    1. Clone repo (if github_url provided) OR use existing path
    2. Load PageIndex JSON tree
    3. Initialize chatbot for code retrieval

    Returns:
        repo_id to use in subsequent queries
    """
    try:
        # Determine repo_id
        if request.repo_id:
            repo_id = request.repo_id
        elif request.github_url:
            repo_id = get_repo_id_from_url(request.github_url)
        else:
            raise HTTPException(400, "Must provide either repo_id or github_url")

        # Get repository path
        if request.github_url:
            if repo_id not in repo_paths:
                print(f"🔄 Cloning repository: {request.github_url}")
                repo_path = clone_github_repo(request.github_url)
                repo_paths[repo_id] = repo_path
            else:
                print(f"✅ Using cached repository: {repo_id}")
                repo_path = repo_paths[repo_id]
        elif request.repo_path:
            repo_path = request.repo_path
            repo_paths[repo_id] = repo_path
        else:
            raise HTTPException(400, "Must provide either github_url or repo_path")

        # Validate paths
        if not Path(repo_path).exists():
            raise HTTPException(404, f"Repository path not found: {repo_path}")
        if not Path(request.json_tree_path).exists():
            raise HTTPException(404, f"JSON tree not found: {request.json_tree_path}")

        # Load PageIndex JSON tree into semantic search
        print(f"\n🔄 Loading PageIndex tree: {request.json_tree_path}")
        semantic_search.load_repository_tree(repo_id, request.json_tree_path)

        # Initialize chatbot for code retrieval
        print(f"\n🔄 Initializing chatbot for: {repo_path}")
        chatbots[repo_id] = ProductionChatbot(repo_path)

        # Get repo info
        repo_info = semantic_search.get_repo_info(repo_id)

        return {
            "status": "success",
            "message": f"Repository '{repo_id}' initialized successfully",
            "repo_id": repo_id,
            "repo_path": repo_path,
            "json_tree_path": request.json_tree_path,
            "stats": repo_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query against a repository.

    Complete flow:
    1. User sends query
    2. Generate query embedding
    3. Search JSON tree using semantic similarity
    4. Retrieve actual code using start_line/end_line
    5. Send to LLM (Ollama) with context
    6. Return LLM response

    This is the MAIN endpoint - simplified to just repo_id + query!
    """
    try:
        # Validate repo is initialized
        if request.repo_id not in semantic_search.list_loaded_repos():
            raise HTTPException(
                400, 
                f"Repository '{request.repo_id}' not initialized. Call /initialize first."
            )

        if request.repo_id not in chatbots:
            raise HTTPException(
                400,
                f"Chatbot for '{request.repo_id}' not initialized. Call /initialize first."
            )

        # STEP 1: Semantic search on PageIndex JSON tree
        print(f"\n🔍 Searching for: '{request.user_query}'")
        filtered_functions = semantic_search.search_and_format_for_chatbot(
            repo_id=request.repo_id,
            query=request.user_query,
            top_k=request.top_k
        )

        print(f"✅ Found {len(filtered_functions)} relevant functions")

        if not filtered_functions:
            return QueryResponse(
                status="success",
                response="I couldn't find any relevant code for your query. Try rephrasing or asking about different functionality.",
                matched_functions=[],
                functions_count=0
            )

        # STEP 2: Retrieve code and generate LLM response
        chatbot = chatbots[request.repo_id]

        print(f"🤖 Generating LLM response...")
        llm_response = chatbot.generate_response(
            user_query=request.user_query,
            filtered_functions=filtered_functions
        )

        print(f"✅ Response generated!")

        return QueryResponse(
            status="success",
            response=llm_response,
            matched_functions=filtered_functions,
            functions_count=len(filtered_functions)
        )

    except HTTPException:
        raise
    except Exception as e:
        return QueryResponse(
            status="error",
            response="",
            matched_functions=[],
            functions_count=0,
            error=str(e)
        )


@app.get("/search/{repo_id}")
async def search_only(repo_id: str, query: str, top_k: int = 5):
    """
    Search endpoint - returns matching functions WITHOUT LLM processing.
    Useful for debugging or building custom workflows.
    """
    try:
        if repo_id not in semantic_search.list_loaded_repos():
            raise HTTPException(400, f"Repository '{repo_id}' not loaded")

        results = semantic_search.search(repo_id, query, top_k)

        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded_repos": semantic_search.list_loaded_repos(),
        "active_chatbots": len(chatbots)
    }


@app.get("/repos")
async def list_repositories():
    """List all initialized repositories with stats."""
    repos = []
    for repo_id in semantic_search.list_loaded_repos():
        info = semantic_search.get_repo_info(repo_id)
        info['has_chatbot'] = repo_id in chatbots
        repos.append(info)

    return {
        "repositories": repos,
        "count": len(repos)
    }


@app.delete("/cleanup/{repo_id}")
async def cleanup_repository(repo_id: str):
    """Clean up a specific repository from memory."""
    removed = []

    if repo_id in chatbots:
        del chatbots[repo_id]
        removed.append("chatbot")

    if repo_id in semantic_search.repositories:
        del semantic_search.repositories[repo_id]
        removed.append("semantic_search")

    if repo_id in repo_paths:
        del repo_paths[repo_id]
        removed.append("repo_path")

    if removed:
        return {
            "status": "success",
            "message": f"Cleaned up {repo_id}",
            "removed": removed
        }
    else:
        return {
            "status": "not_found",
            "message": f"Repository {repo_id} not loaded"
        }


@app.post("/cleanup_all")
async def cleanup_all():
    """Clean up all repositories from memory."""
    count = len(chatbots)

    chatbots.clear()
    repo_paths.clear()
    semantic_search.repositories.clear()

    return {
        "status": "success",
        "message": f"Cleaned up {count} repositories"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)