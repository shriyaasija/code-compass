# 🧭 Code Compass

**AI-powered codebase Q&A** — Ask natural language questions about any GitHub repository and get intelligent, context-aware answers powered by local LLMs.

Code Compass uses a **tree-based semantic search** (PageIndex) to navigate code hierarchies and an LLM to generate detailed answers with exact file/line references.

---

## ✨ Features

- 🌳 **Tree-Based Search** — Hierarchical code navigation using PageIndex JSON trees
- 🤖 **Local LLM Support** — Choose between **Ollama** or **LM Studio** (your data never leaves your machine)
- 🔍 **Smart Code Retrieval** — LLM scores relevance at each tree level, only fetching what matters
- 💬 **Conversational Interface** — Streamlit chat UI for asking questions about codebases
- 📦 **GitHub Integration** — Clone and analyze any public repository
- 🎯 **Needle-in-Haystack Prompting** — Advanced prompting for accurate long-context answers

---

## 📁 Project Structure

```
code-compass/
├── backend/
│   ├── api.py                 # FastAPI server (main entry point)
│   ├── ollama_client.py       # Ollama LLM client
│   ├── lmstudio_client.py     # LM Studio LLM client
│   ├── code_index.py          # Tree-based search engine
│   ├── retrieval.py           # Code retrieval & LLM response generation
│   ├── code_parser.py         # Code parsing utilities
│   └── semantic_search.py     # Semantic search module
├── frontend/
│   └── app.py                 # Streamlit UI
├── mock_repository/           # Sample repo for testing
├── mock_pageindex_tree.json   # Sample PageIndex JSON for demo
├── requirements.txt           # Core dependencies
└── requirements2.txt          # Additional dependencies (tree-sitter)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or **pip**
- An LLM provider (choose one):
  - [Ollama](https://ollama.com/) — lightweight, CLI-based
  - [LM Studio](https://lmstudio.ai/) — GUI-based, easy model management

### 1. Clone the Repository

```bash
git clone https://github.com/shriyaasija/code-compass.git
cd code-compass
```

### 2. Create & Activate Conda Environment

```bash
conda create -n codecompass python=3.11 -y
conda activate codecompass
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements2.txt
```

### 4. Set Up Your LLM Provider

#### Option A: Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull a model (in a separate terminal)
ollama pull qwen3:8b
```

#### Option B: LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Open LM Studio → Search and download a model (e.g., Llama 3.1 8B Instruct)
3. Go to the **Local Server** tab (left sidebar)
4. Click **Start Server** — it runs on `http://localhost:1234`

---

## ▶️ Running the Application

### Step 1: Start the Backend API

**With Ollama (default):**
```bash
conda activate codecompass
python -m backend.api
```

**With LM Studio:**
```bash
conda activate codecompass
LLM_PROVIDER=lmstudio python -m backend.api
```

The API server starts on `http://localhost:8000`.

### Step 2: Start the Frontend (new terminal)

```bash
conda activate codecompass
streamlit run frontend/app.py
```

The Streamlit UI opens at `http://localhost:8501`.

### Step 3: Use the App

1. In the Streamlit sidebar, select your **LLM Provider** (Ollama or LM Studio)
2. Switch to **Demo** mode to test with the included mock repository
3. Set the **Repository Path** to `./mock_repository`
4. Set the **PageIndex JSON Path** to `./mock_pageindex_tree.json`
5. Click **Initialize**
6. Ask questions like:
   - *"How do I train the model?"*
   - *"What is the CNN architecture?"*
   - *"How to preprocess images?"*

---

## ⚙️ Configuration

### Environment Variables

| Variable | Values | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `ollama`, `lmstudio` | `ollama` | Which LLM backend to use |

### Ollama Configuration

- **Default URL:** `http://localhost:11434`
- **Default Model:** `qwen3:8b`
- Change the model in `backend/api.py` line where `OllamaLLM(model=...)` is called

### LM Studio Configuration

- **Default URL:** `http://localhost:1234`
- **Model:** Auto-detected from loaded model in LM Studio
- Just load any model in LM Studio and start the server — Code Compass picks it up automatically

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/initialize` | Load a repository and its PageIndex JSON |
| `POST` | `/query` | Ask a question about the loaded repo |
| `GET` | `/search/{repo_id}?query=...` | Search without LLM response |
| `GET` | `/health` | Health check & provider status |
| `GET` | `/repos` | List loaded repositories |
| `DELETE` | `/cleanup/{repo_id}` | Remove a repo from memory |
| `POST` | `/cleanup_all` | Remove all repos from memory |

### Example API Calls

```bash
# Initialize a repo
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "./mock_repository",
    "json_tree_path": "./mock_pageindex_tree.json",
    "repo_id": "mock_ml_classifier"
  }'

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "mock_ml_classifier",
    "user_query": "How do I train the model?"
  }'

# Check health
curl http://localhost:8000/health
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `ollama serve` fails | Make sure Ollama is installed: `brew install ollama` |
| `ConnectionError: Ollama server not accessible` | Run `ollama serve` in a separate terminal |
| `ConnectionError: LM Studio server not accessible` | Open LM Studio → Local Server → Start Server |
| `streamlit: command not found` | Run with `python -m streamlit run frontend/app.py` |
| `ModuleNotFoundError` | Make sure your conda env is activated: `conda activate codecompass` |
| Empty LLM responses | Try a different/larger model or lower the search threshold |

---

## 🏗️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **LLM Providers:** Ollama / LM Studio (OpenAI-compatible API)
- **Code Parsing:** tree-sitter
- **Search:** Custom tree-based traversal with LLM scoring

---

## 📝 License

This project is for educational and hackathon purposes.
