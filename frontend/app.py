
import streamlit as st 
import requests
import json


API1_URL = "http://localhost:8000"
API2_URL = "http://localhost:8001"


st.set_page_config(
    page_title="Code Compass",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_repository(repo_path=None, json_tree_path=None, github_url=None):
    """Initialize repository in backend API with live polling"""
    import threading
    import time
    
    payload = {"repo_id": "current_repo"}
    if github_url:
        payload["github_url"] = github_url
    if repo_path:
        payload["repo_path"] = repo_path
    if json_tree_path:
        payload["json_tree_path"] = json_tree_path

    result_container = []
    
    def run_request():
        try:
            response = requests.post(
                f"{API1_URL}/initialize",
                json=payload,
                timeout=None
            )
            response.raise_for_status()
            result_container.append(response.json())
        except Exception as e:
            result_container.append({"error": str(e), "status": "error"})
            
    # Start background thread
    t = threading.Thread(target=run_request)
    t.start()
    
    # UI Placeholders
    st.info("🚀 Pinging backend and building code AST... (This takes a moment)")
    progress_ui = st.empty()
    tokens_ui = st.empty()
    
    while t.is_alive():
        try:
            prog_res = requests.get(f"{API1_URL}/initialize_progress", timeout=2)
            if prog_res.status_code == 200:
                prog = prog_res.json()
                with progress_ui.container():
                     status_str = str(prog.get('status', 'unknown')).upper()
                     st.write(f"🔄 **Backend Status**: {status_str}...")
                     
                if prog.get("total_tokens", 0) > 0:
                    with tokens_ui.container():
                        st.markdown("**📝 Live Token Usage (Summarizer)**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Tokens", f"{prog.get('total_tokens', 0):,}")
                        c2.metric("Nodes Summarized", prog.get("nodes_summarized", 0))
                        c3.metric("LLM Calls", prog.get("llm_calls", 0))
        except Exception:
            pass # ignore timeouts on GET
            
        time.sleep(2)
        
    t.join()
    
    # Clear live update placeholders securely before handoff
    progress_ui.empty()
    if 'tokens_ui' in locals():
        tokens_ui.empty()
    
    if result_container:
        if "error" in result_container[0]:
            st.error(f"❌ Initialization failed: {result_container[0]['error']}")
            return None
        return result_container[0]
        
    return None


def call_api_1(repo_id, user_query, top_k=5):
    """Query repository - semantic search happens automatically!"""
    try:
        with st.spinner("🔍 Searching codebase..."):
            response = requests.post(
                f"{API1_URL}/query",
                json={
                    "repo_id": repo_id,
                    "user_query": user_query
                },
                timeout=None  # No timeout for slow local LLMs
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"❌ Query failed: {str(e)}")
        return None

def check_api_health(api_url, name):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_backend_info():
    """Fetch current backend configuration and status"""
    try:
        response = requests.get(f"{API1_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def update_backend_config(provider, model=None):
    """Update backend LLM configuration"""
    try:
        payload = {"provider": provider}
        if model:
            payload["model"] = model
        response = requests.post(f"{API1_URL}/config", json=payload, timeout=30)
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to update config: {e}")
        return None

def main():    
    # Header
    st.markdown('<div class="main-header">Code Compass</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about any GitHub repository using AI</div>', 
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Repository input
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/owner/repo",
            help="Enter the full URL of the GitHub repository"
        )

        # Model selection
        st.subheader("🤖 AI Configuration")
        
        backend_info = get_backend_info()
        current_provider = backend_info.get("llm_provider", "ollama") if backend_info else "ollama"
        current_model = backend_info.get("llm_model", "") if backend_info else ""
        
        provider = st.selectbox(
            "Provider",
            ["ollama", "lmstudio"],
            index=0 if current_provider == "ollama" else 1,
            help="Select the AI service provider"
        )
        
        model_name = st.text_input(
            "Model Name",
            value=current_model,
            placeholder="e.g. qwen3:8b or local-model",
            help="Specific model name. For LM Studio, leave blank to auto-detect."
        )
        
        if st.button("Update AI Model", use_container_width=True):
            res = update_backend_config(provider, model_name if model_name else None)
            if res and res.get("status") == "success":
                st.success(f"Updated to {res.get('model')}")
                st.rerun()
            else:
                st.error("Failed to update model")

        st.markdown("---")

        # API status
        st.subheader("🔌 API Status")

        api1_status = check_api_health(API1_URL, "API1")
        api2_status = check_api_health(API2_URL, "API2")

        if api1_status:
            st.success("API 1: Running")
        else:
            st.error("API 1: Offline")

        if api2_status:
            st.success("API 2: Running")
        else:
            st.warning("API 2: Offline")

        st.markdown("---")

        # Instructions
        st.subheader("How to Use")
        st.markdown("""
        1. Enter a GitHub repository URL
        2. Ask questions about the code
        3. Get instant, intelligent answers!

        **Example Questions:**
        - How does authentication work?
        - Show me the database connection code
        - How do I use the API client?
        - Explain the main function
        """)

        st.markdown("---")

        # About
        with st.expander("About"):
            st.markdown("""
            This chatbot uses:
            - **Ollama** (local LLM)
            - **Mistral/Llama** models
            - **FastAPI** backend
            - **Streamlit** frontend

            No data leaves your machine!
            """)

    # Main content area
    if not repo_url:
        st.info("Enter a GitHub repository URL in the sidebar to get started!")
        return

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_repo" not in st.session_state:
        st.session_state.current_repo = None
    
    if "is_initialized" not in st.session_state:
        st.session_state.is_initialized = False

    # Check if repo changed
    if st.session_state.current_repo != repo_url:
        st.session_state.messages = []
        st.session_state.current_repo = repo_url
        st.session_state.is_initialized = False
        st.info(f"Switched to repository: {repo_url}")

    # Initialization section
    if not st.session_state.is_initialized:
        if st.button("🚀 Initialize Repository", type="primary"):
            if not check_api_health(API1_URL, "API 1"):
                st.error("Your API is not running.")
                return
            result = initialize_repository(github_url=repo_url)
            if result and result.get("status") == "success":
                st.session_state.is_initialized = True
                st.session_state.repo_stats = result
                st.rerun()
        return

    st.success("✅ Repository ready for queries!")
    
    # Check if we just initialized and have stats to show
    if "repo_stats" in st.session_state and st.session_state.repo_stats:
        result = st.session_state.repo_stats
        with st.expander("📊 Extracted Target Repository Analytics", expanded=True):
            sum_tokens = result.get('summarization_tokens', {})
            if sum_tokens.get("total_tokens", 0) > 0:
                st.markdown("**📝 Final Backend Token Usage (Summarizer)**")
                t1, t2, t3 = st.columns(3)
                t1.metric("Total Tokens Processed", f"{sum_tokens.get('total_tokens', 0):,}")
                t2.metric("Nodes Extracted", sum_tokens.get('nodes_summarized', 0))
                t3.metric("LLM API Calls", sum_tokens.get('llm_calls', 0))
            else:
                st.info("Analytics processed offline.")
            
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tokens_used" in message and message["tokens_used"]:
                tokens = message["tokens_used"]
                st.caption(f"*(Tokens: Prompt \u2192 {tokens.get('prompt_tokens', 0):,}, Completion \u2192 {tokens.get('completion_tokens', 0):,}, Total \u2192 {tokens.get('total_tokens', 0):,})*")

    # Chat input
    if prompt := st.chat_input("Ask a question about the code..."):

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):

            # Check API status first
            if not check_api_health(API1_URL, "API 1"):
                st.error("Your API is not running.")
                return

            # Call unified API 1
            your_result = call_api_1(
                repo_id="current_repo",
                user_query=prompt
            )

            if not your_result or your_result.get("status") != "success":
                error_msg = your_result.get("error", "Unknown error") if your_result else "No response"
                st.error(f"Failed to generate response: {error_msg}")
                return

            # Display response
            response_text = your_result.get("response", "")
            st.markdown(response_text)
            
            # Extract and display tokens inline
            tokens_used = your_result.get("tokens_used") or {}
            if tokens_used:
                st.caption(f"*(Tokens: Prompt \u2192 {tokens_used.get('prompt_tokens', 0):,}, Completion \u2192 {tokens_used.get('completion_tokens', 0):,}, Total \u2192 {tokens_used.get('total_tokens', 0):,})*")
            
            # Show stats
            matched_functions = your_result.get("matched_functions", [])
            with st.expander(f"📊 Found {len(matched_functions)} relevant functions"):
                for idx, fn in enumerate(matched_functions):
                    st.code(f"{fn.get('name', 'Unknown')} in {fn.get('file_path', 'Unknown')}")

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "tokens_used": tokens_used
            })


def demo_mode():
    st.title("🧭 Demo Mode - Test Your PageIndex API")

    st.info("This mode tests your semantic search API with the mock repository data.")

    # Repository Configuration
    st.header("📂 Step 1: Repository Configuration")
    col1, col2 = st.columns(2)

    with col1:
        repo_path = st.text_input(
            "Local Repository Path",
            value="./mock_repository",
            placeholder="/path/to/your/repo",
            help="Path to the mock repository folder"
        )

    with col2:
        json_tree_path = st.text_input(
            "PageIndex JSON Path",
            value="./mock_pageindex_tree.json",
            placeholder="/path/to/pageindex.json",
            help="Path to PageIndex JSON tree with embeddings"
        )

    st.divider()

    # Initialize Repository Section
    st.header("🚀 Step 2: Initialize Repository")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Click below to load the repository and PageIndex JSON into the API")
    with col2:
        init_button = st.button("Initialize", type="primary", use_container_width=True)

    if init_button:
        result = initialize_repository(repo_path, json_tree_path)
        if result and result.get('status') == 'success':
            st.success(f"✅ Initialized: {result['repo_id']}")
            st.session_state['repo_id'] = result['repo_id']
            st.session_state['repo_initialized'] = True

            # Show stats
            with st.expander("📊 Repository Statistics", expanded=True):
                stats = result.get('stats', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", stats.get('total_nodes', 0))
                with col2:
                    st.metric("Files", stats.get('node_types', {}).get('file', 0))
                with col3:
                    st.metric("Functions", stats.get('node_types', {}).get('function', 0))

                st.json(stats.get('node_types', {}))
                
                # Show Token stats
                sum_tokens = result.get('summarization_tokens', {})
                if sum_tokens.get("total_tokens", 0) > 0:
                    st.divider()
                    st.markdown("**📝 Summarization Token Usage**")
                    t1, t2, t3 = st.columns(3)
                    with t1:
                        st.metric("Total Tokens", f"{sum_tokens.get('total_tokens', 0):,}")
                    with t2:
                        st.metric("Nodes Summarized", sum_tokens.get('nodes_summarized', 0))
                    with t3:
                        st.metric("LLM API Calls", sum_tokens.get('llm_calls', 0))
        else:
            st.error("❌ Failed to initialize repository")
            if result:
                st.json(result)

    # Show initialization status
    if st.session_state.get('repo_initialized'):
        st.success(f"✅ Repository ready: {st.session_state.get('repo_id')}")
    else:
        st.warning("⚠️ Repository not initialized yet")

    st.divider()

    # Query Section
    st.header("❓ Step 3: Ask Questions")

    # Test query suggestions
    st.markdown("**Try these example queries:**")
    example_queries = [
        "How do I train the model?",
        "How to preprocess images?",
        "What is the CNN architecture?",
        "How to save checkpoints?",
        "Calculate accuracy metrics"
    ]

    cols = st.columns(len(example_queries))
    selected_example = None
    for i, (col, query) in enumerate(zip(cols, example_queries)):
        with col:
            if st.button(f"📝 {i+1}", key=f"example_{i}", help=query):
                selected_example = query
                st.session_state['user_query'] = query  # Store in session state

    # Use session state for query if available
    default_query = st.session_state.get('user_query', selected_example if selected_example else "")

    user_query = st.text_area(
        "Your Question",
        value=default_query,
        placeholder="e.g., How do I train the model?",
        help="Ask anything about the codebase",
        height=100,
        key="query_input"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of results", 1, 10, 5, help="How many relevant functions to retrieve")
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        query_button = st.button("🔍 Search", type="primary", use_container_width=True)

    st.divider()

    # ========================================================================
    # CRITICAL: DISPLAY RESULTS SECTION - THIS MUST BE OUTSIDE BUTTON HANDLER
    # ========================================================================

    # Show results if available
    if 'last_search_result' in st.session_state and st.session_state['last_search_result']:
        st.markdown("### 🎯 Search Results")
        result = st.session_state['last_search_result']

        # Debug: Show what we got
        st.write(f"DEBUG: Result status = {result.get('status')}")
        st.write(f"DEBUG: Number of matched functions = {len(result.get('matched_functions', []))}")

        if result.get("status") == "success":
            st.success("✅ Search Complete!")

            matched_functions = result.get('matched_functions', [])

            if matched_functions:
                st.markdown(f"### 📝 Found {len(matched_functions)} Relevant Functions")

                for i, func in enumerate(matched_functions, 1):
                    relevance = func.get('relevance_score', 0)

                    if relevance > 0.8:
                        icon = "🟢"
                    elif relevance > 0.6:
                        icon = "🟡"
                    else:
                        icon = "🟠"

                    with st.expander(
                        f"{icon} {i}. **{func.get('name', 'Unknown')}** - Relevance: {relevance:.3f}",
                        expanded=(i == 1)
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.code(func.get('signature', 'No signature'), language='python')

                        with col2:
                            st.markdown(f"**File:** `{func.get('file_path', 'N/A')}`")
                            st.markdown(f"**Lines:** {func.get('start_line', 0)}-{func.get('end_line', 0)}")

                        if func.get('docstring'):
                            st.markdown("**Description:**")
                            st.info(func['docstring'])

                # Show LLM response if available
                llm_response = result.get('response', '')
                if llm_response and llm_response.strip():
                    st.divider()
                    st.markdown("### 💬 AI Explanation")
                    st.markdown(llm_response)
                else:
                    st.info("💡 LLM response not available. Install Ollama for AI explanations.")
                
                # Token tracking display
                tokens_used = result.get('tokens_used', {})
                if tokens_used and tokens_used.get('total_tokens', 0) > 0:
                    st.divider()
                    st.markdown("**(Token Usage for AI Explanation)**")
                    tk1, tk2, tk3 = st.columns(3)
                    with tk1:
                        st.metric("Prompt Tokens", f"{tokens_used.get('prompt_tokens', 0):,}")
                    with tk2:
                        st.metric("Completion Tokens", f"{tokens_used.get('completion_tokens', 0):,}")
                    with tk3:
                        st.metric("Total Tokens", f"{tokens_used.get('total_tokens', 0):,}")
                        
            else:
                st.warning("⚠️ No matching functions found. Try a different query.")
        else:
            st.error("❌ Search Failed")
            st.write("Full response:")
            st.json(result)

    # ========================================================================
    # BUTTON HANDLER - STORES RESULT AND TRIGGERS RERUN
    # ========================================================================

    if query_button:
        if not user_query or not user_query.strip():
            st.error("⚠️ Please enter a question!")
            st.stop()

        if not st.session_state.get('repo_initialized'):
            st.error("⚠️ Please initialize the repository first!")
            st.stop()

        if not check_api_health(API1_URL, "API"):
            st.error("❌ API is not running on port 8000.")
            st.code("cd backend\npython api_updated.py", language="bash")
            st.stop()

        with st.spinner("🔍 Searching codebase..."):
            try:
                # Call API
                result = call_api_1(
                    repo_id=st.session_state.get('repo_id'),
                    user_query=user_query,
                    top_k=top_k
                )

                st.write("DEBUG: Got result from API")
                st.write(f"DEBUG: Result type = {type(result)}")
                st.write(f"DEBUG: Result = {result}")

                # Store in session state
                st.session_state['last_search_result'] = result

                # Trigger rerun
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error calling API: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Debug section
    with st.expander("🔧 Debug Info"):
        st.markdown("**Session State:**")
        st.json({
            "repo_id": st.session_state.get('repo_id', 'Not set'),
            "repo_initialized": st.session_state.get('repo_initialized', False),
            "has_last_result": 'last_search_result' in st.session_state,
            "user_query": st.session_state.get('user_query', 'Not set')
        })

        st.markdown("**API Health:**")
        api_healthy = check_api_health(API1_URL, 'API1')
        st.write(f"API 1 (port 8000): {'✅ Running' if api_healthy else '❌ Offline'}")

        if st.button("Test API Health", key="test_health"):
            try:
                response = requests.get(f"{API1_URL}/health", timeout=5)
                st.success("✅ API is reachable")
                st.json(response.json())
            except Exception as e:
                st.error(f"❌ Cannot reach API: {str(e)}")

        if st.button("Clear Results", key="clear_results"):
            if 'last_search_result' in st.session_state:
                del st.session_state['last_search_result']
            st.success("Cleared!")
            st.rerun()

if __name__ == "__main__":
    # Mode selection
    mode = st.sidebar.radio(
        "🎮 Mode",
        ["Production", "Demo"],
        help="Production: Full integration | Demo: Test your API only"
    )

    if mode == "Demo":
        demo_mode()
    else:
        main()