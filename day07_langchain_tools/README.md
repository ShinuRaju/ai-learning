# Day 7 - Tool Calling with LangChain

**Learning Goal:** Build an AI agent that can use multiple tools through LangChain's tool calling system.

## 🎯 What You'll Learn

- **LangChain Tools**: How to create and use tools in LangChain
- **Agent + Tool Pattern**: Agents that decide which tools to use
- **Tool Abstraction**: Clean separation between tool logic and AI decision-making
- **Multiple Tool Coordination**: Agent managing several different capabilities

## 🔧 Tools We'll Build

1. **Calculator Tool**: Perform mathematical calculations
2. **DateTime Tool**: Get current date and time information
3. **File Reader Tool**: Read and summarize text files

## 📁 Project Structure

```
day07_langchain_tools/
├── tool_agent.py          # Main agent with tools
├── README.md             # This file
└── requirements.txt      # Dependencies
```

## 🚀 How to Run

```bash
cd day07_langchain_tools
pip install -r requirements.txt
python tool_agent.py
```

## 💡 Key Concepts

### vs Day 4 (Manual Tool Simulation):

- **Day 4**: We manually decided when to use tools and parsed responses
- **Day 7**: LangChain handles tool calling automatically with function calling

### vs Day 6 (Basic Chains):

- **Day 6**: Simple prompt → LLM → response chain
- **Day 7**: LLM can decide to call tools and use their results

## 🔄 How Tool Calling Works

```python
# User asks: "What's 15 * 23 and what time is it?"

# Agent thinks: "I need calculator for math and datetime for time"

# Step 1: Call calculator tool
calculator_result = calculator.run("15 * 23")  # Returns "345"

# Step 2: Call datetime tool
time_result = datetime.run()  # Returns "2025-12-30 14:30:00"

# Step 3: Combine results
# Agent responds: "15 * 23 = 345, and it's currently 2:30 PM on December 30, 2025"
```

## 🎓 Learning Progression

**Day 4** → **Day 6** → **Day 7**
Manual Tools → LangChain Chains → LangChain Tools + Agents

You're building the foundation for sophisticated AI agents! 🤖✨
