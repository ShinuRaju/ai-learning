# Day 6 - LangChain Basics: LangChain Chatbot

## 🎯 Learning Goals

- **Chains**: Understanding LangChain's core abstraction for combining components
- **Prompt templates**: Reusable, structured prompts vs. raw strings
- **LangChain patterns**: How LangChain simplifies LLM interactions
- **Component composition**: Building workflows with | operator

## 🔍 Project Overview

Rebuild the Day 1 CLI chatbot using LangChain abstractions to demonstrate how frameworks simplify AI application development. This project shows the same functionality with cleaner, more maintainable code.

## 📋 Key Concepts

### 🔗 **LangChain Chains**

The core abstraction that connects components together:

```python
chain = prompt_template | llm | output_parser
# Input flows through: template → LLM → parser → output
```

### 📝 **Prompt Templates**

Structured, reusable prompts instead of string formatting:

```python
# Old way (Day 1)
prompt = f"System: You are helpful\nUser: {user_input}"

# LangChain way (Day 6)
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    ("human", "{user_input}")
])
```

### 🎨 **Component Composition**

Easy to swap and modify components:

```python
# Same LLM, different prompts
creative_chain = creative_prompt | llm | parser
technical_chain = technical_prompt | llm | parser
```

## 🚀 How to Run

1. **Install LangChain dependencies:**

   ```bash
   cd day06_langchain_basics
   pip install -r requirements.txt
   ```

2. **Set up your API key:**

   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   # On Windows: set OPENROUTER_API_KEY=your-api-key-here
   ```

3. **Run the LangChain chatbot:**

   ```bash
   python langchain_chatbot.py
   ```

4. **Try these features:**
   - Normal conversation (same as Day 1, but with LangChain)
   - `/demo` - See LangChain concepts demonstration
   - Compare creative vs technical assistant responses

## 🔧 Key Technical Differences

### Day 1 (Raw API) vs Day 6 (LangChain)

#### **Day 1 Approach:**

```python
payload = {
    "model": "mistralai/devstral-2512:free",
    "messages": [
        {"role": "system", "content": "You are helpful..."},
        {"role": "user", "content": user_input}
    ]
}
response = requests.post(url, headers=headers, json=payload)
result = response.json()
return result['choices'][0]['message']['content']
```

#### **Day 6 Approach:**

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful..."),
    ("human", "{user_input}")
])
chain = prompt_template | llm | output_parser
response = chain.invoke({"user_input": user_input})
```

## 📚 What You'll Learn

### Core LangChain Concepts:

1. **LLM Abstractions**: Unified interface for different language models
2. **Prompt Templates**: Parameterized prompts with variables
3. **Chains**: Composable workflows that connect components
4. **Output Parsers**: Transform raw LLM output into structured data
5. **Pipe Operator (|)**: LangChain's elegant chain composition syntax

### Key Benefits Discovered:

- **Cleaner Code**: Less boilerplate, more readable
- **Reusability**: Templates and chains can be reused
- **Maintainability**: Easy to modify and extend
- **Error Handling**: Built-in robust error handling
- **Consistency**: Standard patterns across different models
- **Ecosystem**: Rich library of pre-built components

## 🎓 LangChain Architecture

```
User Input
     ↓
Prompt Template (formats input with system prompt)
     ↓
LLM (processes formatted prompt)
     ↓
Output Parser (formats response)
     ↓
Final Response
```

## 🔍 Interactive Demo Features

### `/demo` Command Shows:

1. **Prompt Template Structure**: How templates work vs. raw strings
2. **Chain Composition**: Visual representation of component flow
3. **Different Personalities**: Same question, different prompt styles
4. **Template Flexibility**: Easy to create specialized assistants

### Example Demo Output:

```
Question: "How do you learn programming?"

🎨 Creative Assistant:
→ "Embark on a coding adventure! Start with curiosity as your compass..."

🔧 Technical Assistant:
→ "Begin with fundamentals: variables, functions, control structures..."
```

## 🎯 Extensions to Try

1. **Custom Output Parsers**: Parse responses into JSON or structured data
2. **Memory Integration**: Add conversation memory to chains
3. **Multiple LLM Comparison**: Same chain with different models
4. **Complex Chains**: Multi-step reasoning chains
5. **Prompt Libraries**: Create a collection of reusable templates

## 🔗 Progression to Next Days

Day 6 establishes the LangChain foundation for:

- **Day 7**: Tool calling with LangChain agents
- **Day 8**: Multi-step reasoning chains
- **Day 9**: Advanced memory strategies
- **Day 10**: Complex agent workflows

## ⚡ Performance & Simplicity

### Lines of Code Comparison:

- **Day 1 (Raw API)**: ~15 lines for basic LLM call
- **Day 6 (LangChain)**: ~3 lines for same functionality

### Readability Score:

- **Day 1**: Technical implementation details visible
- **Day 6**: High-level abstractions, intent-focused

## 🛠️ Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure `langchain` and `langchain-openai` are installed
2. **API Configuration**: LangChain uses OpenAI-compatible interface
3. **Model Names**: Ensure correct model name for OpenRouter

### Debug Mode:

Enable LangChain debug logging to see chain execution:

```python
import langchain
langchain.debug = True
```

## 💡 Key Insight

LangChain doesn't change **what** you can do, but makes it **much easier** to do it well. The same functionality from Day 1 becomes more maintainable, reusable, and extensible.

**Mental Model**: Think of LangChain as high-level programming for AI applications - you focus on **what** you want to accomplish, not **how** to make individual API calls.

---

_LangChain transforms AI development from plumbing to architecture! 🏗️_
