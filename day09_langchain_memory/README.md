# Day 9 - LangChain Memory: Conversational Agent

Build a conversational AI agent that remembers previous exchanges and maintains context throughout the conversation.

## 🎯 Learning Goals

- Understand LangChain memory types and patterns
- Implement conversation buffer memory management
- Learn memory persistence and retrieval strategies
- Analyze cost vs context tradeoffs
- Build a conversational agent with persistent memory

## 🛠️ What We're Building

A conversational agent that:

1. **Remembers conversations** - Maintains context across exchanges
2. **Multiple memory strategies** - Buffer, Window, and Summary memory
3. **Cost optimization** - Understand token usage and memory limits
4. **Memory management** - Save, clear, and analyze conversation history

## 🧠 Key Concepts

### Memory Types

- **Buffer Memory**: Keeps complete conversation history
- **Window Memory**: Maintains only recent exchanges (sliding window)
- **Summary Memory**: Compresses old conversations into summaries
- **Summary Buffer**: Combines summary and recent buffer

### Cost Considerations

- **Token Growth**: How memory affects API costs
- **Context Limits**: Managing maximum context windows
- **Memory Strategies**: Choosing the right approach for your use case

## 🔧 Installation

```bash
pip install langchain langchain-openai
```

## 🚀 Usage

```bash
python memory_agent.py
```

### Interactive Commands

- Type normally to chat with the agent
- `memory` - Show current memory information
- `clear` - Clear conversation memory
- `save` - Save conversation to file
- `quit` - Exit the program

## 📊 Memory Type Comparison

| Type    | Pros             | Cons              | Use Case                |
| ------- | ---------------- | ----------------- | ----------------------- |
| Buffer  | Complete history | Growing costs     | Short conversations     |
| Window  | Fixed cost       | Loses old context | Long conversations      |
| Summary | Scales well      | May lose details  | Very long conversations |

## 💡 Key Features

### 1. Conversation Context

```python
# The agent remembers previous exchanges
User: "My name is Alice"
AI: "Nice to meet you, Alice!"

User: "What's my name?"
AI: "Your name is Alice!"
```

### 2. Memory Management

```python
# Show memory information
agent.show_memory_info()

# Clear memory
agent.clear_memory()

# Save conversation
agent.save_conversation()
```

### 3. Multiple Memory Strategies

```python
# Different memory types
agent = ConversationalAgent(memory_type="buffer")      # Full history
agent = ConversationalAgent(memory_type="buffer_window") # Last 5 exchanges
agent = ConversationalAgent(memory_type="summary")     # Compressed history
```

## 🔍 How It Works

### 1. Memory Setup

```python
def setup_memory(self):
    if self.memory_type == "buffer":
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )
```

### 2. Conversation Chain

```python
def setup_conversation_chain(self):
    self.prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with memory..."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
```

### 3. Memory Integration

```python
def chat(self, user_input: str) -> str:
    response = self.conversation_chain.invoke({"input": user_input})

    # Save exchange to memory
    self.memory.save_context(
        {"input": user_input},
        {"output": response}
    )
    return response
```

## 🎮 Try These Conversations

1. **Personal Information**:

   - "My name is [Your Name] and I'm learning AI"
   - Ask later: "What do you know about me?"

2. **Multi-turn Planning**:

   - "I want to plan a vacation to Japan"
   - "What should I pack for a 2-week trip?"
   - "Actually, make it 1 week instead"

3. **Technical Discussion**:
   - "Explain neural networks to me"
   - "How does that relate to transformers?"
   - "Give me a code example"

## 🔬 Experiments to Try

1. **Memory Types**: Test different memory strategies with the same conversation
2. **Memory Limits**: See how window memory behaves with long conversations
3. **Cost Analysis**: Compare token usage across memory types
4. **Persistence**: Save and load conversations across sessions

## 📈 Next Steps

- Day 10: Mini use case with expense explainer agent
- Combine memory with tools and reasoning
- Learn RAG (Retrieval-Augmented Generation)
- Build more sophisticated agents

## 🔗 Connection to Previous Days

- **Day 7**: Tool usage patterns → Now with memory
- **Day 8**: Multi-step reasoning → Now with conversation context
- **Future**: RAG systems will extend this memory concept

---

_Remember: The key insight is understanding the tradeoff between memory completeness and cost. Choose your memory strategy based on your specific use case!_
