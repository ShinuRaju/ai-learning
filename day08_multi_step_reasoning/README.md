# Day 8 - Multi-Step Reasoning

**Learning Goal:** Build an AI agent that can break down complex problems into steps and reason through them systematically.

## 🎯 What You'll Learn

- **Think → Plan → Act Pattern**: Systematic problem-solving approach
- **Reasoning Loops**: How agents iterate through complex problems
- **Agent Scratchpad**: Maintaining reasoning state across steps
- **Step-by-Step Problem Decomposition**: Breaking complex tasks into manageable pieces

## 🧠 Multi-Step Reasoning Concept

Instead of direct responses, the agent will:

1. **🤔 THINK**: Analyze the problem and identify what needs to be done
2. **📋 PLAN**: Create a step-by-step approach
3. **⚡ ACT**: Execute the plan using available tools
4. **🔄 REFLECT**: Review results and continue if needed

## 📁 Project Structure

```
day08_multi_step_reasoning/
├── reasoning_agent.py     # Main multi-step reasoning agent
├── README.md             # This file
└── requirements.txt      # Dependencies
```

## 🚀 How to Run

```bash
cd day08_multi_step_reasoning
pip install -r requirements.txt
python reasoning_agent.py
```

## 💡 Key Concepts

### vs Day 7 (Direct Tool Calling):

- **Day 7**: User question → Tool decision → Single tool use → Response
- **Day 8**: User question → Think → Plan → Multiple coordinated actions → Response

### Reasoning Loop Example:

```
User: "Calculate the area of a circle with radius 5, then tell me what percentage that is of a square with side length 10"

🤔 THINK: I need to:
  - Calculate circle area (π × r²)
  - Calculate square area (side²)
  - Calculate percentage
  - Present the results

📋 PLAN:
  Step 1: Calculate circle area using calculator tool
  Step 2: Calculate square area using calculator tool
  Step 3: Calculate percentage using calculator tool
  Step 4: Present comprehensive answer

⚡ ACT:
  Step 1: calculator("3.14159 * 5 * 5") → 78.54
  Step 2: calculator("10 * 10") → 100
  Step 3: calculator("78.54 / 100 * 100") → 78.54%

🔄 REFLECT: All calculations complete, can provide final answer
```

## 🔄 How Reasoning Loops Work

1. **Scratchpad**: Agent maintains running notes of its thinking
2. **Iterative Process**: Can take multiple reasoning cycles
3. **Tool Coordination**: Uses multiple tools in sequence
4. **Self-Direction**: Decides when the task is complete

## 🎓 Learning Progression

**Day 7** → **Day 8**
Single Tool Use → Multi-Step Coordination
Direct Response → Planned Execution
Simple Logic → Complex Reasoning

You're building the foundation for advanced AI reasoning! 🚀✨
