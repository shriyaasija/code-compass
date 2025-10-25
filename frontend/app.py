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

def call_api_1(repo_id, repo_path, user_query, 
                  filtered_functions, model):
    try:
        with st.spinner("Retrieving code and generating answer..."):
            response = requests.post(
                f"{API1_URL}/query",
                json={
                    "repo_id": repo_id,
                    "repo_path": repo_path,
                    "user_query": user_query,
                    "filtered_functions": filtered_functions,
                    "model": model
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to chatbot API. Is it running on port 8000?")
        return None
    except requests.exceptions.Timeout:
        st.error("Response generation timed out. Try a simpler query.")
        return None
    except Exception as e:
        st.error(f"Chatbot API error: {str(e)}")
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
    st.title("Demo Mode - Test Your API")
        
    col1, col2 = st.columns(2)
    
    with col1:
        repo_path = st.text_input(
            "Local Repository Path",
            placeholder="/path/to/your/repo",
            help="Path to a local repository for testing"
        )
    
    with col2:
        model = st.selectbox(
            "Model",
            ["qwen3:8b"]
        )
    
    user_query = st.text_area(
        "Your Question",
        placeholder="How does authentication work?"
    )
    
    # Mock function input
    st.subheader("Mock Filtered Functions")
    st.caption("In production, these come from partners' API")
    
    mock_functions_json = st.text_area(
        "Filtered Functions (JSON)",
        value=json.dumps([
            {
                "name": "example_function",
                "signature": "def example_function(x, y)",
                "file_path": "src/main.py",
                "start_line": 10,
                "end_line": 25,
                "docstring": "Example function"
            }
        ], indent=2),
        height=200
    )
    
    if st.button("Test Your API", type="primary"):
        try:
            mock_functions = json.loads(mock_functions_json)
            
            result = call_api_1(
                repo_id="demo_repo",
                repo_path=repo_path,
                user_query=user_query,
                filtered_functions=mock_functions,
                model=model
            )
            
            if result and result.get("status") == "success":
                st.success("API Response Received")
                st.markdown("### Response:")
                st.markdown(result.get("response"))
            else:
                st.error("API Error")
                if result:
                    st.json(result)
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON in filtered functions")
        except Exception as e:
            st.error(f"Error: {str(e)}")

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