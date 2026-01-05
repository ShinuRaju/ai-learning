# Day 1 - Hello LLM: CLI AI Assistant

## 🎯 Learning Objectives

- Understand LLM API basics
- Learn request/response flow
- Build a simple CLI interface

## 🚀 Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenRouter API key (optional):**

   ```bash
   # Option 1: Environment variable (recommended)
   export OPENROUTER_API_KEY="your-api-key-here"

   # Option 2: Windows PowerShell
   $env:OPENROUTER_API_KEY = "your-api-key-here"
   ```

   _Note: A default API key is already included in the code for this demo_

3. **Run the assistant:**
   ```bash
   python cli_assistant.py
   ```

## 💡 Key Concepts Learned

### 1. LLM API Basics

- **API Key Authentication**: Secure way to authenticate with OpenRouter
- **Model Selection**: Using `mistralai/devstral-2512:free` for cost efficiency
- **Message Structure**: System and user roles in conversation
- **HTTP Requests**: Direct API calls using the requests library

### 2. Request/Response Flow

```
User Input → API Request → LLM Processing → API Response → Display Output
```

### 3. Error Handling

- API key validation
- Network error handling
- User input validation

## 🔍 Code Walkthrough

### Core Components:

1. **CLIAssistant Class**: Encapsulates the LLM interaction logic
2. **get_llm_response()**: Handles the API call to OpenAI
3. **run()**: Main CLI loop for user interaction

### API Call Structure:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant..."},
        {"role": "user", "content": user_input}
    ],
    max_tokens=150,
    temperature=0.7
)
```

## 🎮 Try These Commands

- "What's the weather like today?"
- "Explain quantum computing in simple terms"
- "Write a haiku about programming"
- "Help me debug this Python error: NameError"

## 🔄 Next Steps (Day 2)

Tomorrow we'll learn about **prompt control** and building a role-based assistant!

## 💡 Reflection Questions

1. How does the system prompt affect the assistant's behavior?
2. What happens when you change the `temperature` parameter?
3. How could you modify this to save conversation history?
