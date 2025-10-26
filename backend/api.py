from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
import tempfile
import subprocess
from pathlib import Path

from backend.retrieval import ProductionChatbot
from backend.code_index import TreeBasedSearch
from backend.ollama_client import OllamaLLM

app = FastAPI(title="CodeCompass - Tree-Based Search Edition")

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
tree_search = None  # Will be initialized on startup
llm_client = None  # Ollama client

# Initialize LLM and tree search on startup
@app.on_event("startup")
async def startup_event():
    global tree_search, llm_client
    
    try:
        print("\n" + "="*70)
        print("🚀 INITIALIZING CODE COMPASS API")
        print("="*70)
        
        # Initialize Ollama client
        print("\n📡 Connecting to Ollama...")
        llm_client = OllamaLLM(model="qwen3:8b")  # Using your model
        print("✅ Ollama client initialized")
        
        # Initialize tree search with LLM client
        print("\n🌳 Initializing tree-based search...")
        tree_search = TreeBasedSearch(llm_client=llm_client, threshold=0.5)
        print("✅ Tree search initialized with threshold=0.5")
        
        print("\n" + "="*70)
        print("✅ API READY TO ACCEPT REQUESTS")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ STARTUP FAILED: {e}")
        print("Make sure Ollama is running: ollama serve")
        raise


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
    repo_id: Optional[str] = None
    threshold: Optional[float] = 0.5  # Tree search threshold


class QueryRequest(BaseModel):
    """Simplified query request - just repo_id and query!"""
    repo_id: str
    user_query: str
    threshold: Optional[float] = None  # Optional per-query threshold override


class QueryResponse(BaseModel):
    """Response with LLM-generated answer."""
    status: str
    response: str
    matched_functions: List[dict]  # All leaf nodes that were matched
    functions_count: int
    error: Optional[str] = None


@app.post("/initialize")
async def initialize_repository(request: InitializeRequest):
    """
    Initialize a repository for querying with tree-based search.

    Steps:
    1. Clone repo (if github_url provided) OR use existing path
    2. Load PageIndex JSON tree
    3. Initialize chatbot for code retrieval

    Returns:
        repo_id to use in subsequent queries
    """
    try:
        # Check if tree_search is initialized
        if tree_search is None:
            raise HTTPException(
                500, 
                "Tree search not initialized. API startup may have failed."
            )
        
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

        # Update threshold if provided
        if request.threshold is not None:
            tree_search.threshold = request.threshold

        # Load PageIndex JSON tree into tree search
        print(f"\n🔄 Loading PageIndex tree: {request.json_tree_path}")
        tree_search.load_repository_tree(repo_id, request.json_tree_path)

        # Initialize chatbot for code retrieval
        print(f"\n🔄 Initializing chatbot for: {repo_path}")
        chatbots[repo_id] = ProductionChatbot(repo_path)

        # Get repo info
        repo_info = tree_search.get_repo_info(repo_id)

        return {
            "status": "success",
            "message": f"Repository '{repo_id}' initialized successfully",
            "repo_id": repo_id,
            "repo_path": repo_path,
            "json_tree_path": request.json_tree_path,
            "threshold": tree_search.threshold,
            "stats": repo_info
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Initialize error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query against a repository.

    Complete flow:
    1. User sends query
    2. LLM traverses tree scoring relevance at each level
    3. Returns ALL leaf nodes found above threshold
    4. Retrieve actual code using start_line/end_line
    5. Send to LLM (Ollama) with context
    6. Return LLM response

    This returns ALL relevant functions, not a fixed top_k!
    """
    try:
        # Validate repo is initialized
        if request.repo_id not in tree_search.list_loaded_repos():
            raise HTTPException(
                400, 
                f"Repository '{request.repo_id}' not initialized. Call /initialize first."
            )

        if request.repo_id not in chatbots:
            raise HTTPException(
                400,
                f"Chatbot for '{request.repo_id}' not initialized. Call /initialize first."
            )

        # Update threshold for this query if provided
        original_threshold = tree_search.threshold
        if request.threshold is not None:
            tree_search.threshold = request.threshold
            print(f"🔧 Using custom threshold: {request.threshold}")

        try:
            # STEP 1: Tree-based search - returns ALL leaf nodes found
            print(f"\n🔍 Searching for: '{request.user_query}'")
            filtered_functions = tree_search.search_and_format_for_chatbot(
                repo_id=request.repo_id,
                query=request.user_query
            )

            print(f"✅ Found {len(filtered_functions)} relevant leaf nodes")

            if not filtered_functions:
                return QueryResponse(
                    status="success",
                    response="I couldn't find any relevant code for your query. Try lowering the threshold or rephrasing your question.",
                    matched_functions=[],
                    functions_count=0
                )

            # STEP 2: Retrieve code and generate LLM response
            chatbot = chatbots[request.repo_id]

            print(f"🤖 Generating LLM response with {len(filtered_functions)} functions...")
            llm_response = chatbot.generate_response(
                user_query=request.user_query,
                filtered_functions=filtered_functions
            )

            print(f"✅ Response generated!")
            print(llm_response)

            return QueryResponse(
                status="success",
                response=llm_response,
                matched_functions=filtered_functions,
                functions_count=len(filtered_functions)
            )
        
        finally:
            # Restore original threshold
            if request.threshold is not None:
                tree_search.threshold = original_threshold

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Query error: {str(e)}")
        print(traceback.format_exc())
        return QueryResponse(
            status="error",
            response="",
            matched_functions=[],
            functions_count=0,
            error=str(e)
        )


@app.get("/search/{repo_id}")
async def search_only(repo_id: str, query: str, threshold: float = None):
    """
    Search endpoint - returns ALL matching leaf nodes WITHOUT LLM processing.
    Useful for debugging or building custom workflows.
    """
    try:
        if repo_id not in tree_search.list_loaded_repos():
            raise HTTPException(400, f"Repository '{repo_id}' not loaded")

        # Use custom threshold if provided
        original_threshold = tree_search.threshold
        if threshold is not None:
            tree_search.threshold = threshold

        try:
            results = tree_search.search(repo_id, query)

            return {
                "status": "success",
                "query": query,
                "threshold": tree_search.threshold,
                "results": results,
                "count": len(results)
            }
        finally:
            if threshold is not None:
                tree_search.threshold = original_threshold

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ollama_status = "connected" if llm_client else "not initialized"
    
    return {
        "status": "healthy",
        "search_type": "tree_based",
        "ollama_status": ollama_status,
        "ollama_model": llm_client.model if llm_client else None,
        "loaded_repos": tree_search.list_loaded_repos() if tree_search else [],
        "active_chatbots": len(chatbots),
        "threshold": tree_search.threshold if tree_search else None
    }

@app.get("/repos")
async def list_repositories():
    """List all initialized repositories with stats."""
    if not tree_search:
        return {"repositories": [], "count": 0}
    
    repos = []
    for repo_id in tree_search.list_loaded_repos():
        info = tree_search.get_repo_info(repo_id)
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

    if tree_search and repo_id in tree_search.repositories:
        del tree_search.repositories[repo_id]
        removed.append("tree_search")

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
    if tree_search:
        tree_search.repositories.clear()

    return {
        "status": "success",
        "message": f"Cleaned up {count} repositories"
    }


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting CodeCompass API Server...")
    print("📡 Make sure Ollama is running: ollama serve")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)