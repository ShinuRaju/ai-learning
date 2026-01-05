#!/usr/bin/env python3
"""
Day 5 - Memory Lite: Chat with Short-term Memory

This script demonstrates how to build a chat system with short-term memory that:
- Stores the last 5 messages (user + assistant pairs)
- Injects conversation context into prompts
- Shows how memory affects AI responses
- Demonstrates context window limitations

Key Learning Goals:
- Context windows and their limitations
- Memory management strategies
- How conversation history affects responses
- Trade-offs between memory length and cost
"""

import json
import os
import sys
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

class ConversationMemory:
    """Manages short-term memory for chat conversations."""
    
    def __init__(self, max_messages: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of message pairs to remember
        """
        self.max_messages = max_messages
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(message)
        
        # Keep only the last max_messages * 2 messages (user + assistant pairs)
        max_total_messages = self.max_messages * 2
        if len(self.conversation_history) > max_total_messages:
            # Remove oldest messages but try to keep complete pairs
            messages_to_remove = len(self.conversation_history) - max_total_messages
            self.conversation_history = self.conversation_history[messages_to_remove:]
    
    def get_conversation_context(self) -> str:
        """
        Get conversation history formatted as context for the LLM.
        
        Returns:
            Formatted conversation history
        """
        if not self.conversation_history:
            return "No previous conversation."
        
        context_lines = ["Previous conversation:"]
        for msg in self.conversation_history:
            role = "You" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")
        
        return "\\n".join(context_lines)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage."""
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,  # Rough estimate: 4 chars per token
            "max_messages_limit": self.max_messages
        }
    
    def clear_memory(self):
        """Clear all conversation history."""
        self.conversation_history = []
    
    def export_conversation(self) -> List[Dict[str, str]]:
        """Export full conversation history."""
        return self.conversation_history.copy()

class MemoryEnabledChat:
    """AI chat assistant with short-term memory capabilities."""
    
    def __init__(self, memory_size: int = 5):
        """Initialize the memory-enabled chat assistant."""
        self.api_key = os.getenv('OPENROUTER_API_KEY') or "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"
        if not self.api_key:
            print("⚠️  Warning: OPENROUTER_API_KEY not found in environment variables")
            print("Please set it using: export OPENROUTER_API_KEY='your-api-key'")
            sys.exit(1)
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-learning-project",
            "X-Title": "AI Learning Day 5 - Memory Lite"
        }
        
        self.memory = ConversationMemory(max_messages=memory_size)
    
    def create_contextual_prompt(self, user_message: str) -> str:
        """
        Create a prompt that includes conversation context.
        
        Args:
            user_message: The current user message
            
        Returns:
            Full prompt with conversation context
        """
        conversation_context = self.memory.get_conversation_context()
        
        system_prompt = """You are a helpful AI assistant with memory. You can reference previous parts of our conversation to provide more relevant and personalized responses.

Pay attention to:
- Names and personal details mentioned earlier
- Topics discussed previously
- Questions asked before
- Context that might be relevant to the current question

Be natural and conversational. Reference previous conversation when it's helpful, but don't force it."""

        # Combine system prompt, conversation context, and current message
        full_prompt = f"""{system_prompt}

{conversation_context}

Current message: {user_message}

Please respond helpfully, using the conversation context when relevant."""

        return full_prompt
    
    def get_ai_response(self, user_message: str) -> Dict[str, Any]:
        """
        Get AI response with conversation context.
        
        Args:
            user_message: User's input message
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Create contextual prompt
            prompt = self.create_contextual_prompt(user_message)
            
            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,  # Slightly higher for more conversational responses
                "max_tokens": 300
            }
            
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            response_data = response.json()
            assistant_response = response_data['choices'][0]['message']['content'].strip()
            
            # Add both user message and assistant response to memory
            self.memory.add_message("user", user_message)
            self.memory.add_message("assistant", assistant_response)
            
            return {
                "success": True,
                "response": assistant_response,
                "memory_stats": self.memory.get_memory_stats(),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Error getting response: {str(e)}",
                "memory_stats": self.memory.get_memory_stats(),
                "error": str(e)
            }
    
    def show_memory_status(self):
        """Display current memory status."""
        stats = self.memory.get_memory_stats()
        print("\\n📊 Memory Status:")
        print(f"   💬 Total messages: {stats['total_messages']}")
        print(f"   👤 User messages: {stats['user_messages']}")
        print(f"   🤖 Assistant messages: {stats['assistant_messages']}")
        print(f"   📝 Total characters: {stats['total_characters']}")
        print(f"   🎫 Estimated tokens: {stats['estimated_tokens']}")
        print(f"   🧠 Memory limit: {stats['max_messages_limit']} message pairs")
        
        if stats['total_messages'] >= stats['max_messages_limit'] * 2:
            print("   ⚠️  Memory is at limit - oldest messages being forgotten")
    
    def export_conversation(self, filename: Optional[str] = None):
        """Export conversation to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.json"
        
        conversation = self.memory.export_conversation()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            print(f"✅ Conversation exported to: {filename}")
        except Exception as e:
            print(f"❌ Failed to export conversation: {e}")
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear_memory()
        print("🗑️  Memory cleared!")

def main():
    """Main interactive loop for the memory-enabled chat."""
    print("🧠 Memory-Enabled Chat - Day 5: Memory Lite")
    print("=" * 55)
    print("Chat with an AI that remembers your conversation!")
    print("The AI will remember the last 5 message pairs.")
    print("\\nSpecial commands:")
    print("  /memory  - Show memory status")
    print("  /clear   - Clear conversation memory")
    print("  /export  - Export conversation to file")
    print("  /quit    - Exit the chat")
    print("\\nTry having a multi-turn conversation to see memory in action!")
    print("Examples:")
    print("  - Tell the AI your name, then ask questions later")
    print("  - Discuss a topic across multiple messages")
    print("  - Ask 'What did I just tell you?' after sharing information")
    print("\\n" + "=" * 55)
    
    # Create chat assistant with 5-message memory
    chat = MemoryEnabledChat(memory_size=5)
    
    # Demonstration conversation
    print("\\n🔍 Let's see how memory works with a quick demo:")
    demo_exchanges = [
        ("Hi, my name is Alex and I'm learning about AI!", None),
        ("What programming languages are best for AI?", None),
        ("Do you remember my name?", "Notice how it remembers!")
    ]
    
    for user_msg, note in demo_exchanges:
        print(f"\\n👤 You: {user_msg}")
        result = chat.get_ai_response(user_msg)
        print(f"🤖 Assistant: {result['response']}")
        if note:
            print(f"💡 {note}")
    
    chat.show_memory_status()
    
    print("\\n" + "=" * 55)
    print("Now try your own conversation:")
    
    while True:
        try:
            user_input = input("\\n👤 You: ").strip()
            
            if not user_input:
                print("Please enter a message.")
                continue
            
            # Handle special commands
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            elif user_input.lower() == '/memory':
                chat.show_memory_status()
                continue
            elif user_input.lower() == '/clear':
                chat.clear_memory()
                continue
            elif user_input.lower() == '/export':
                chat.export_conversation()
                continue
            
            # Get AI response with memory
            result = chat.get_ai_response(user_input)
            print(f"🤖 Assistant: {result['response']}")
            
            # Show memory info if it's getting full
            stats = result['memory_stats']
            if stats['total_messages'] >= 8:  # Close to limit
                print(f"\\n💭 Memory: {stats['total_messages']}/10 messages, {stats['estimated_tokens']} tokens")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()