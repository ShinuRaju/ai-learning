# Day 4 - Tool Simulation: Calculator Assistant

## 🎯 Learning Goals

- **Tool-calling concept (manual)**: Understanding how LLMs can decide when to use external tools
- **Decision making**: Teaching AI when tools are needed vs. direct responses
- **Execution separation**: Separating reasoning (LLM) from execution (Python)
- **Tool orchestration**: Manual coordination between AI decisions and tool execution

## 🔍 Project Overview

Build a calculator assistant that demonstrates the foundation of tool-calling systems. The LLM analyzes user queries and decides whether mathematical calculations are needed. If so, it specifies which tools to use, and Python executes the calculations.

## 📋 Features

### Core Tools

- 🧮 **calculate()**: Basic math expressions (e.g., "2 + 3 \* 4")
- √ **square_root()**: Square root calculations
- ^ **power()**: Exponentiation (e.g., "2^8")

### Intelligence Pipeline

1. **Query Analysis**: LLM analyzes if calculations are needed
2. **Tool Selection**: AI specifies which tools and parameters to use
3. **Execution**: Python safely executes the mathematical operations
4. **Response Generation**: LLM creates natural language response with results

### Example Interactions

```
👤 User: "What is 25 * 8 + 12?"
🤖 Assistant: Let me calculate that for you: 25 * 8 + 12 = 212

👤 User: "What's the capital of France?"
🤖 Assistant: The capital of France is Paris. (No calculations needed)

👤 User: "What's the square root of 144?"
🤖 Assistant: The square root of 144 is 12.0
```

## 🚀 How to Run

1. **Set up your environment:**

   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   # On Windows: set OPENROUTER_API_KEY=your-api-key-here
   ```

2. **Install dependencies:**

   ```bash
   cd day04_tool_simulation
   pip install -r requirements.txt
   ```

3. **Run the calculator assistant:**

   ```bash
   python calculator_assistant.py
   ```

4. **Try these examples:**
   - "Calculate 15 \* 23 + 47"
   - "What's 2 to the power of 10?"
   - "Find the square root of 256"
   - "What is the meaning of life?" (no tools needed)
   - "If I buy 7 items at $12.50 each, how much total?"

## 🔧 Key Technical Concepts

### 1. Tool Decision Making

```python
def create_tool_decision_prompt(self, user_query: str) -> str:
    return f"""You are a helpful assistant with access to calculator tools.

    Analyze: "{user_query}"

    Respond with JSON:
    - If NO calculation needed: {{"needs_calculation": false, "response": "answer"}}
    - If calculation needed: {{"needs_calculation": true, "tools_to_use": [...]}}
    """
```

### 2. Safe Tool Execution

```python
@staticmethod
def calculate(expression: str) -> Dict[str, Any]:
    # Remove dangerous characters for safety
    safe_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
    result = eval(safe_expression)  # Safe after sanitization
```

### 3. Structured Tool Pipeline

1. **Decision Phase**: LLM decides if tools needed
2. **Execution Phase**: Python runs calculations
3. **Response Phase**: LLM formats final answer

## 📚 What You'll Learn

### Day 4 Focus Areas:

1. **Tool Abstraction**: Creating reusable tool functions
2. **AI Decision Making**: Teaching LLMs when to use tools
3. **Safety Patterns**: Sanitizing inputs for safe execution
4. **Manual Orchestration**: Coordinating AI reasoning and tool execution
5. **Structured Communication**: JSON protocols for tool requests/responses

### Key Insights:

- **Separation of Concerns**: AI thinks, Python executes
- **Decision Points**: Not every query needs tools
- **Safety First**: Always sanitize tool inputs
- **Structured Interfaces**: JSON makes tool calling predictable
- **Error Handling**: Tools can fail - handle gracefully

## 🎓 Extensions to Try

1. **More Tools**: Add date/time, file operations, web search
2. **Tool Chaining**: Multiple tools in sequence for complex calculations
3. **Parameter Validation**: Stronger input validation for tools
4. **Tool Discovery**: Let AI learn about available tools dynamically
5. **Cost Tracking**: Monitor API calls and tool usage

## 🔗 Progression to Next Days

This manual tool simulation is the foundation for:

- **Day 7**: LangChain's automatic tool calling
- **Day 16+**: Agent tool systems
- **Day 26+**: MCP tool protocols

Understanding manual tool orchestration helps you:

- Debug automatic tool systems
- Design custom tool integrations
- Build reliable agent workflows
- Understand tool safety patterns

## 🛡️ Safety Notes

- Input sanitization prevents code injection
- Limited tool scope reduces attack surface
- Error handling prevents crashes
- Safe evaluation with regex filtering

## 💡 Mental Model

```
User Query → LLM Decision → Tool Execution → LLM Response
     ↓            ↓              ↓             ↓
"Calculate     needs_calc:     Python runs   "The answer
 2 + 3"        true, tool:      eval("2+3")   is 5"
               "calculate"
```

---

_Manual tool simulation teaches you the fundamentals of AI tool orchestration! 🔧_
