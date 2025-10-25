
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
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
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


def initialize_repository(repo_path, json_tree_path):
    """Initialize repository in backend API"""
    try:
        with st.spinner("🔄 Initializing repository..."):
            response = requests.post(
                f"{API1_URL}/initialize",
                json={
                    "repo_path": repo_path,
                    "json_tree_path": json_tree_path,
                    "repo_id": "mock_ml_classifier"
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None


def call_api_1(repo_id, user_query, top_k=5):
    """Query repository - semantic search happens automatically!"""
    try:
        with st.spinner("🔍 Searching codebase..."):
            response = requests.post(
                f"{API1_URL}/query",
                json={
                    "repo_id": repo_id,
                    "user_query": user_query,
                    "top_k": top_k
                },
                timeout=120
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
        model = st.selectbox(
            "AI Model",
            ["qwen3:8b"],
            help="Select the Ollama model to use"
        )

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

    # Check if repo changed
    if st.session_state.current_repo != repo_url:
        st.session_state.messages = []
        st.session_state.current_repo = repo_url
        st.info(f"Switched to repository: {repo_url}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

            # Step 1: Call partners' API
            partners_result = None

            if not partners_result:
                st.error("Failed to analyze repository")
                return

            filtered_functions = partners_result.get("filtered_functions", [])

            if not filtered_functions:
                st.warning("No relevant functions found for your query. Try rephrasing.")
                return

            st.caption(f"Found {len(filtered_functions)} relevant functions")

            # Step 2: Call your API
            your_result = call_api_1(
                repo_id=partners_result.get("repo_id"),
                repo_path=partners_result.get("repo_path"),
                user_query=prompt,
                filtered_functions=filtered_functions,
                model=model
            )

            if not your_result or your_result.get("status") != "success":
                error_msg = your_result.get("error", "Unknown error") if your_result else "No response"
                st.error(f"Failed to generate response: {error_msg}")
                return

            # Display response
            response_text = your_result.get("response", "")
            st.markdown(response_text)

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })

            # Show stats
            st.caption(f"Analyzed {your_result.get('functions_count', 0)} functions")


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