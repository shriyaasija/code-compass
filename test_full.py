import ollama
def get_functions_context(functions_list):
    """
    Extract function signatures and source code to use as context
    """
    context_parts = []
    
    for func in functions_list:
        # Get function signature
        signature = str(inspect.signature(func))
        # Get function source code
        source = inspect.getsource(func)
        # Get docstring
        docstring = inspect.getdoc(func) or "No description"
        
        context_parts.append(f"""
Function: {func.__name__}{signature}
Description: {docstring}
Source Code:
{source}
""")
    
    return "\n".join(context_parts)

# Step 4: Create the system prompt with functions as context
def chat_with_function_context(user_prompt: str, functions_array: list):
    """
    Send a prompt to Qwen3 with functions as reference context
    
    Args:
        user_prompt: The user's question
        functions_array: List of function references to include as context
    """
    # Generate context from functions
    functions_context = get_functions_context(functions_array)
    
    # Create system message with function context
    system_message = f"""You are a helpful assistant with access to the following Python functions for reference:

{functions_context}

You can reference these functions when answering questions. These are provided as context to help you understand what operations are available."""
    
    # Send chat request with context
    response = ollama.chat(
        model='qwen3:8b',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    
    return response['message']['content']

# Step 5: Alternative - Send function context as formatted text
def chat_with_formatted_context(user_prompt: str, functions_array: list):
    """
    Alternative approach: send functions as formatted context without source code
    """
    # Create simplified function descriptions
    function_descriptions = []
    for func in functions_array:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description"
        function_descriptions.append(
            f"- {func.__name__}{sig}: {doc}"
        )
    
    context_text = "Available functions:\n" + "\n".join(function_descriptions)
    
    # Combine context with user prompt
    full_prompt = f"""{context_text}

User Question: {user_prompt}"""
    
    response = ollama.chat(
        model='qwen3:8b',
        messages=[
            {'role': 'user', 'content': full_prompt}
        ]
    )
    
    return response['message']['content']

# Example Usage
if __name__ == "__main__":
    # Example 1: Using system message with full function context
    print("=" * 60)
    print("Example 1: Full function context in system message")
    print("=" * 60)
    
    user_question = "I have a rectangle with length 10 and width 5. What operations can I perform on it?"
    
    answer = chat_with_function_context(user_question, functions_array)
    print(f"Question: {user_question}")
    print(f"Answer: {answer}\n")
    
    # Example 2: Using simplified context in user prompt
    print("=" * 60)
    print("Example 2: Simplified context in prompt")
    print("=" * 60)
    
    user_question2 = "How do I convert 98.6 Fahrenheit to Celsius using the available functions?"
    
    answer2 = chat_with_formatted_context(user_question2, functions_array)
    print(f"Question: {user_question2}")
    print(f"Answer: {answer2}\n")
    
    # Example 3: Conversational with maintained context
    print("=" * 60)
    print("Example 3: Conversation with persistent context")
    print("=" * 60)
    
    # Build conversation with context
    functions_context = get_functions_context(functions_array)
    system_prompt = f"You are a helpful assistant. You have access to these functions:\n{functions_context}"
    
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]
    
    # First question
    messages.append({'role': 'user', 'content': 'What functions do you have access to?'})
    response1 = ollama.chat(model='qwen3:8b', messages=messages)
    print(f"User: What functions do you have access to?")
    print(f"Assistant: {response1['message']['content']}\n")
    
    # Add assistant response to conversation
    messages.append(response1['message'])
    
    # Second question
    messages.append({'role': 'user', 'content': 'Calculate the area of a 15x20 rectangle'})
    response2 = ollama.chat(model='qwen3:8b', messages=messages)
    print(f"User: Calculate the area of a 15x20 rectangle")
    print(f"Assistant: {response2['message']['content']}")