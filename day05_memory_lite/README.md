# Day 5 - Memory Lite: Chat with Short-term Memory

## 🎯 Learning Goals

- **Context windows**: Understanding how conversation context affects AI responses
- **Memory limitations**: Why we need to manage conversation history length
- **Memory strategies**: Simple approaches to conversation memory
- **Cost vs context trade-offs**: Balancing memory length with API costs

## 🔍 Project Overview

Build a chat system with short-term memory that remembers the last 5 message pairs. The AI can reference previous parts of the conversation to provide more contextual and personalized responses.

## 📋 Features

### Memory Management

- 🧠 **5-Message Memory**: Remembers last 5 user/assistant message pairs
- 🔄 **Automatic Rotation**: Oldest messages are forgotten when limit is reached
- 📊 **Memory Stats**: Track message count, character usage, and token estimates
- 🗑️ **Memory Clear**: Option to reset conversation history

### Conversation Context

- 💬 **Contextual Responses**: AI references previous conversation when relevant
- 👤 **Personal Memory**: Remembers names, preferences, and topics discussed
- 🔗 **Topic Continuity**: Can follow multi-turn conversations naturally
- ⚡ **Smart Context**: Only includes relevant conversation history in prompts

### Utilities

- 📁 **Export Conversations**: Save chat history to JSON files
- 📈 **Token Estimation**: Track approximate token usage for cost awareness
- 🎛️ **Interactive Commands**: Special commands for memory management

## 🚀 How to Run

1. **Set up your environment:**

   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   # On Windows: set OPENROUTER_API_KEY=your-api-key-here
   ```

2. **Install dependencies:**

   ```bash
   cd day05_memory_lite
   pip install -r requirements.txt
   ```

3. **Run the memory-enabled chat:**

   ```bash
   python memory_chat.py
   ```

4. **Try these conversation patterns:**
   - **Personal Context**: "Hi, I'm Sarah and I love cooking" → later ask "What do you remember about me?"
   - **Topic Continuation**: Discuss a project across multiple messages
   - **References**: "What did I just tell you?" after sharing information
   - **Multi-turn Planning**: Plan something step-by-step across several exchanges

## 🔧 Key Technical Concepts

### 1. Conversation Memory Structure

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages  # Message pairs to remember
        self.conversation_history = []    # List of message objects

    def add_message(self, role: str, content: str):
        # Add message and maintain size limit
```

### 2. Context Injection

```python
def create_contextual_prompt(self, user_message: str) -> str:
    conversation_context = self.memory.get_conversation_context()

    full_prompt = f"""
    {system_prompt}
    {conversation_context}
    Current message: {user_message}
    """
```

### 3. Memory Rotation Strategy

- Keep last N message pairs (user + assistant)
- Remove oldest messages when limit exceeded
- Preserve conversation flow by maintaining pairs

### 4. Token Management

```python
def get_memory_stats(self) -> Dict[str, Any]:
    total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
    return {
        "estimated_tokens": total_chars // 4,  # ~4 chars per token
        "total_messages": len(self.conversation_history)
    }
```

## 📚 What You'll Learn

### Day 5 Focus Areas:

1. **Context Windows**: How conversation history affects AI behavior
2. **Memory Trade-offs**: Balancing context length vs. cost/performance
3. **Message Management**: Strategies for rotating conversation history
4. **Token Awareness**: Understanding cost implications of long contexts
5. **User Experience**: How memory improves conversational flow

### Key Insights:

- **Memory Improves Coherence**: AI can follow threads across multiple turns
- **Context Has Limits**: Both technical (token limits) and practical (cost)
- **Rotation Strategy Matters**: How you manage memory affects conversation quality
- **Token Cost Scales**: Longer conversations = higher API costs
- **Personalization**: Memory enables more personalized interactions

## 💡 Interactive Commands

While chatting, you can use these special commands:

- `/memory` - Show current memory status and statistics
- `/clear` - Clear all conversation memory (fresh start)
- `/export` - Export conversation history to JSON file
- `/quit` - Exit the chat application

## 🎓 Extensions to Try

1. **Adaptive Memory**: Adjust memory size based on conversation complexity
2. **Selective Memory**: Remember only important parts of conversations
3. **Persistent Memory**: Save/load conversation history across sessions
4. **Memory Summarization**: Compress old conversations into summaries
5. **Memory Search**: Find specific information from conversation history

## 🔗 Progression to Next Days

This memory foundation prepares you for:

- **Day 6**: LangChain's memory abstractions
- **Day 9**: Advanced memory strategies in LangChain
- **Day 11-15**: RAG systems (external memory)
- **Day 16+**: Agent memory and state management

## 📊 Memory vs. Cost Analysis

| Memory Size | Avg Tokens | API Cost  | Use Case              |
| ----------- | ---------- | --------- | --------------------- |
| 2 pairs     | ~200       | Low       | Simple Q&A            |
| 5 pairs     | ~500       | Medium    | **This project**      |
| 10 pairs    | ~1000      | High      | Complex conversations |
| 20 pairs    | ~2000      | Very High | Document discussions  |

## 💭 Understanding Context Windows

### What Happens Without Memory:

```
👤: Hi, I'm Alex
🤖: Nice to meet you!
👤: What's my name?
🤖: I don't have information about your name
```

### With Memory (Day 5):

```
👤: Hi, I'm Alex
🤖: Nice to meet you, Alex!
👤: What's my name?
🤖: Your name is Alex, as you mentioned when we started talking
```

## 🛡️ Memory Limitations

- **Token Limits**: Most models have context limits (4k, 8k, 32k tokens)
- **Cost Scaling**: Longer contexts = more expensive API calls
- **Relevance Decay**: Old context may become less relevant
- **Privacy Concerns**: Persistent memory raises data storage questions

---

_Building conversation memory is essential for creating natural, coherent AI interactions! 🧠💬_
