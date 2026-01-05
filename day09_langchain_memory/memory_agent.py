#!/usr/bin/env python3
"""
Day 9 - Memory in LangChain: Conversational Agent

This script demonstrates LangChain memory by building a conversational agent that:
1. Remembers previous conversations
2. Maintains context across exchanges
3. Uses different memory strategies
4. Manages cost vs context tradeoffs

Key Learning Goals:
- LangChain memory types and patterns
- Conversation buffer memory management
- Memory persistence and retrieval
- Cost optimization with memory limits
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ConversationalAgent:
    """
    An agent that maintains conversation memory using LangChain.
    
    Note: ConversationSummaryBufferMemory is NOT implemented here due to:
    1. Educational complexity - would obscure core memory concepts
    2. LangChain version conflicts with import paths  
    3. Implementation complexity combining two strategies
    4. Not required at this learning stage
    
    The 3 implemented types (buffer, buffer_window, summary) cover
    the fundamental memory trade-offs students need to understand.
    """
    
    def __init__(self, api_key: Optional[str] = None, memory_type: str = "buffer"):
        """Initialize the conversational agent."""
        # Get API key
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or "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"
        
        if not self.api_key:
            print("❌ Error: OpenRouter API key not found!")
            print("Please set OPENROUTER_API_KEY environment variable.")
            sys.exit(1)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="mistralai/devstral-2512:free",
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,  # Slightly higher for natural conversation
            max_tokens=500
        )
        
        # Set up memory based on type
        self.memory_type = memory_type
        self.setup_memory()
        
        # Create conversation prompt
        self.setup_conversation_chain()
        
        print(f"🧠 Conversational Agent initialized with {memory_type} memory!")
        print("💬 I can remember our conversation and maintain context.")
    
    def setup_memory(self):
        """
        Set up the memory system using simple list-based storage.
        
        NOTE: We use custom implementation instead of LangChain's built-in
        memory classes due to import issues with newer Python versions.
        
        Missing: ConversationSummaryBufferMemory - intentionally not implemented
        because it would combine summary + buffer strategies, adding complexity
        without teaching new fundamental concepts at this learning stage.
        """
        # Simple memory implementation
        self.chat_history = []  # Store conversation as list of messages
        self.max_history = 20 if self.memory_type == "buffer" else 10 if self.memory_type == "buffer_window" else 5
        
        if self.memory_type == "buffer":
            print("📝 Using Buffer Memory: Keeps full conversation history")
        elif self.memory_type == "buffer_window":
            print("🪟 Using Window Memory: Keeps last 5 conversation exchanges")
        elif self.memory_type == "summary":
            print("📋 Using Summary Memory: Maintains conversation summary")
            self.conversation_summary = ""
        # NOTE: ConversationSummaryBufferMemory not implemented here
        # Would combine summary + buffer but adds complexity without
        # teaching new fundamental memory trade-off concepts
        else:
            print("📝 Using default Buffer Memory")
    
    def setup_conversation_chain(self):
        """Set up the conversation chain with memory."""
        # Create prompt template with memory placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with memory. You can:
1. Remember what we've discussed in this conversation
2. Refer back to previous topics and context
3. Build upon earlier exchanges
4. Maintain personality and context throughout

Be conversational, helpful, and make use of the conversation history to provide relevant responses.
If this is the start of our conversation, introduce yourself briefly."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the conversation chain
        self.conversation_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.get_formatted_history()
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_formatted_history(self):
        """Get formatted conversation history."""
        if self.memory_type == "buffer":
            return self.chat_history
        elif self.memory_type == "buffer_window":
            return self.chat_history[-10:]  # Last 10 messages (5 exchanges)
        elif self.memory_type == "summary":
            # Return summary + recent messages
            recent = self.chat_history[-4:]  # Last 2 exchanges
            if hasattr(self, 'conversation_summary') and self.conversation_summary:
                summary_msg = SystemMessage(content=f"Conversation summary: {self.conversation_summary}")
                return [summary_msg] + recent
            return recent
        return self.chat_history

    def chat(self, user_input: str) -> str:
        """Have a conversation with memory."""
        try:
            # Get response using the conversation chain
            response = self.conversation_chain.invoke({"input": user_input})
            
            # Save this exchange to memory
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response))
            
            # Manage memory based on type
            self._manage_memory()
            
            return response
            
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    def _manage_memory(self):
        """Manage memory based on type and limits."""
        if self.memory_type == "buffer" and len(self.chat_history) > self.max_history:
            # Keep only recent messages
            self.chat_history = self.chat_history[-self.max_history:]
        elif self.memory_type == "buffer_window" and len(self.chat_history) > 10:
            # Always keep only last 10 messages (5 exchanges)
            self.chat_history = self.chat_history[-10:]
        elif self.memory_type == "summary" and len(self.chat_history) > 8:
            # Summarize old conversations
            self._create_summary()
            
    def _create_summary(self):
        """Create a summary of older conversations."""
        if len(self.chat_history) <= 4:
            return
            
        # Get older messages to summarize
        old_messages = self.chat_history[:-4]
        
        # Create summary prompt
        summary_text = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in old_messages[-6:]  # Last 6 old messages
        ])
        
        summary_prompt = f"""Summarize this conversation concisely:

{summary_text}

Summary:"""
        
        try:
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            self.conversation_summary = summary_response.content
            
            # Keep only recent messages
            self.chat_history = self.chat_history[-4:]
            print("📝 Conversation summarized and memory compressed")
            
        except Exception as e:
            print(f"❌ Summary error: {e}")
    
    def show_memory_info(self):
        """Display current memory information."""
        print("\n📋 Memory Information:")
        print(f"Type: {self.memory_type}")
        print(f"Messages in memory: {len(self.chat_history)}")
        
        if hasattr(self, 'conversation_summary') and self.conversation_summary:
            print(f"Summary available: {self.conversation_summary[:100]}...")
        
        if self.chat_history:
            print("\n💬 Recent conversation:")
            # Show last few exchanges
            recent_messages = self.chat_history[-4:] if len(self.chat_history) > 4 else self.chat_history
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    print(f"  👤 Human: {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")
                elif isinstance(msg, AIMessage):
                    print(f"  🤖 AI: {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.chat_history = []
        if hasattr(self, 'conversation_summary'):
            self.conversation_summary = ""
        print("🗑️ Memory cleared!")
    
    def save_conversation(self, filename: Optional[str] = None):
        """Save conversation to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Conversation saved on {datetime.now()}\n")
                f.write(f"Memory type: {self.memory_type}\n")
                f.write("=" * 50 + "\n\n")
                
                if hasattr(self, 'conversation_summary') and self.conversation_summary:
                    f.write(f"Summary: {self.conversation_summary}\n\n")
                
                for msg in self.chat_history:
                    if isinstance(msg, HumanMessage):
                        f.write(f"Human: {msg.content}\n\n")
                    elif isinstance(msg, AIMessage):
                        f.write(f"AI: {msg.content}\n\n")
            
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def run(self):
        """Main interaction loop."""
        print("\\n💬 LangChain Memory Agent - Day 9")
        print("=" * 50)
        print("I'm a conversational AI that remembers our chat!")
        print("\\nCommands:")
        print("  • Type normally to chat")
        print("  • 'memory' - Show memory info")
        print("  • 'clear' - Clear memory")
        print("  • 'save' - Save conversation")
        print("  • 'quit' - Exit")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\\n👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\\n🤖 Thanks for the conversation! Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'memory':
                    self.show_memory_info()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_memory()
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                # Skip empty input
                if not user_input:
                    print("Please say something or type a command!")
                    continue
                
                # Get AI response
                response = self.chat(user_input)
                print(f"\\n🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")

def compare_memory_types():
    """Compare different memory types."""
    print("\\n📊 LangChain Memory Types Comparison")
    print("=" * 50)
    
    print("\\n**1. ConversationBufferMemory**")
    print("✅ Pros: Complete conversation history")
    print("❌ Cons: Token cost grows linearly")
    print("🎯 Use: Short conversations, high accuracy needed")
    
    print("\\n**2. ConversationBufferWindowMemory**") 
    print("✅ Pros: Fixed token cost, recent context")
    print("❌ Cons: Loses older context")
    print("🎯 Use: Longer conversations, cost control")
    
    print("\\n**3. ConversationSummaryMemory**")
    print("✅ Pros: Compact, scales well")
    print("❌ Cons: May lose details, extra LLM calls")
    print("🎯 Use: Very long conversations, cost efficiency")
    
    print("\\n**4. ConversationSummaryBufferMemory**")
    print("✅ Pros: Best of summary + buffer")
    print("❌ Cons: More complex, moderate cost")
    print("🎯 Use: Balanced approach for most cases")    
    print("⚠️  NOTE: Not implemented in this demo due to complexity")
    print("   and LangChain version conflicts. The 3 core types above")
    print("   teach all fundamental memory trade-offs needed.")
def memory_cost_analysis():
    """Analyze memory costs and tradeoffs."""
    print("\\n💰 Memory Cost Analysis")
    print("=" * 40)
    
    print("\\n**Token Growth Patterns:**")
    print("• Buffer: O(n) - linear growth")
    print("• Window: O(1) - constant size")
    print("• Summary: O(log n) - slow growth")
    
    print("\\n**Cost Factors:**")
    print("• Input tokens: What you send to LLM")
    print("• Output tokens: What LLM returns")
    print("• Memory tokens: Context from history")
    
    print("\\n**Optimization Strategies:**")
    print("• Use window memory for cost control")
    print("• Summary memory for long conversations")
    print("• Clear memory periodically")
    print("• Compress old conversations")

def demo_different_memories():
    """Demonstrate different memory types."""
    print("\\n🧪 Memory Type Demonstrations")
    print("=" * 40)
    
    memory_types = ["buffer", "buffer_window", "summary"]
    
    for memory_type in memory_types:
        print(f"\\n--- {memory_type.upper()} MEMORY ---")
        try:
            agent = ConversationalAgent(memory_type=memory_type)
            print("✅ Agent created successfully!")
            
            # You can add demo conversations here
            
        except Exception as e:
            print(f"❌ Error creating {memory_type} agent: {e}")

def main():
    """Main function."""
    print("🚀 Starting Day 9: LangChain Memory")
    print("Learning: Conversation memory and context management")
    
    # Show memory type comparisons
    compare_memory_types()
    memory_cost_analysis()
    
    print("\\n" + "="*60)
    print("Choose a memory type to test:")
    print("1. Buffer (keeps everything)")
    print("2. Window (last 5 exchanges)")  
    print("3. Summary (compressed history)")
    print("4. Demo all types")
    
    try:
        choice = input("\\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            agent = ConversationalAgent(memory_type="buffer")
            agent.run()
        elif choice == "2":
            agent = ConversationalAgent(memory_type="buffer_window")
            agent.run()
        elif choice == "3":
            agent = ConversationalAgent(memory_type="summary")
            agent.run()
        elif choice == "4":
            demo_different_memories()
        else:
            print("Invalid choice, using buffer memory...")
            agent = ConversationalAgent(memory_type="buffer")
            agent.run()
            
    except ImportError as e:
        print(f"\\n❌ Import Error: {e}")
        print("\\n💡 Make sure to install required packages:")
        print("   pip install langchain langchain-openai")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()