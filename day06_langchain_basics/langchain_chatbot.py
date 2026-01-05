#!/usr/bin/env python3
"""
Day 6 - LangChain Basics: LangChain Chatbot

This script demonstrates LangChain fundamentals by rebuilding the Day 1 chatbot
using LangChain abstractions instead of raw API calls.

Key Learning Goals:
- LangChain basic concepts (Chains, LLMs, PromptTemplates)
- How LangChain simplifies LLM interactions
- Prompt templates vs. raw strings
- Chain composition and reusability
"""

import os
import sys
from typing import Optional

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser

class SimpleOutputParser(BaseOutputParser):
    """Simple parser that returns the content as-is."""
    
    def parse(self, text: str) -> str:
        return text.strip()

class LangChainChatbot:
    """A chatbot built with LangChain abstractions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LangChain chatbot."""
        # Get API key
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or "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"
        
        if not self.api_key:
            print("❌ Error: OpenRouter API key not found!")
            print("Please set OPENROUTER_API_KEY environment variable.")
            sys.exit(1)
        
        # Initialize LangChain LLM with OpenRouter
        self.llm = ChatOpenAI(
            model="mistralai/devstral-2512:free",
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=300
        )
        
        # Create a prompt template - key LangChain concept
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be friendly, concise, and helpful. Answer questions clearly and provide useful information."),
            ("human", "{user_input}")
        ])
        
        # Create a chain - combines prompt template + LLM
        self.chain = self.prompt_template | self.llm | SimpleOutputParser()
        
        print("🎯 LangChain chatbot initialized successfully!")
    
    def get_response(self, user_input: str) -> str:
        """
        Get AI response using LangChain chain.
        
        Args:
            user_input: User's message
            
        Returns:
            AI response string
        """
        try:
            # Use the chain to process input
            response = self.chain.invoke({"user_input": user_input})
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def demonstrate_langchain_concepts(self):
        """Demonstrate key LangChain concepts."""
        print("\\n🔍 LangChain Concepts Demonstration")
        print("=" * 50)
        
        # 1. Show prompt template
        print("\\n1️⃣ **Prompt Template**:")
        print("Instead of manually formatting strings, LangChain uses templates:")
        print(f"Template: {self.prompt_template}")
        
        # 2. Show chain composition
        print("\\n2️⃣ **Chain Composition**:")
        print("prompt_template | llm | output_parser")
        print("   ↓               ↓         ↓")
        print("Template input → LLM call → Parsed output")
        
        # 3. Demonstrate different prompt styles
        print("\\n3️⃣ **Different Prompt Templates**:")
        
        # Create a few different prompt templates
        creative_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a creative writing assistant. Be imaginative and inspiring."),
            ("human", "{user_input}")
        ])
        
        technical_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical expert. Provide precise, factual answers."),
            ("human", "{user_input}")
        ])
        
        # Create chains with different prompts
        creative_chain = creative_prompt | self.llm | SimpleOutputParser()
        technical_chain = technical_prompt | self.llm | SimpleOutputParser()
        
        test_question = "How do you learn programming?"
        
        print(f"\\nQuestion: '{test_question}'\\n")
        
        try:
            print("🎨 Creative Assistant Response:")
            creative_response = creative_chain.invoke({"user_input": test_question})
            print(f"→ {creative_response}\\n")
            
            print("🔧 Technical Assistant Response:")
            technical_response = technical_chain.invoke({"user_input": test_question})
            print(f"→ {technical_response}\\n")
            
        except Exception as e:
            print(f"Demo error: {e}")
        
        print("💡 Notice how the same question gets different styles of answers!")
        print("   This shows the power of prompt templates in LangChain.")
    
    def run(self):
        """Main chat loop."""
        print("\\n🤖 LangChain Chatbot - Day 6: LangChain Basics")
        print("=" * 55)
        print("This chatbot uses LangChain abstractions instead of raw API calls!")
        print("\\nType 'quit' to exit, '/demo' for LangChain concepts demo")
        print("-" * 55)
        
        while True:
            try:
                # Get user input
                user_input = input("\\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\\n🤖 Goodbye! Thanks for exploring LangChain! 👋")
                    break
                
                if user_input.lower() == '/demo':
                    self.demonstrate_langchain_concepts()
                    continue
                
                # Skip empty input
                if not user_input:
                    print("Please enter a message or type 'quit' to exit.")
                    continue
                
                print("\\n🤖 Assistant: ", end="")
                
                # Get response using LangChain
                response = self.get_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye! Thanks for exploring LangChain!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")

def compare_with_day1():
    """Show comparison between Day 1 (raw API) and Day 6 (LangChain) approaches."""
    print("\\n📊 Day 1 vs Day 6 Comparison")
    print("=" * 40)
    print("\\n**Day 1 (Raw API)**:")
    print("""
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
    """)
    
    print("\\n**Day 6 (LangChain)**:")
    print("""
    prompt_template = ChatPromptTemplate.from_messages([...])
    chain = prompt_template | llm | output_parser
    response = chain.invoke({"user_input": user_input})
    """)
    
    print("\\n✅ **LangChain Benefits**:")
    print("  • Cleaner, more readable code")
    print("  • Reusable prompt templates")
    print("  • Built-in error handling")
    print("  • Easy chain composition")
    print("  • Consistent patterns across models")
    print("  • Rich ecosystem of components")

def main():
    """Main function."""
    print("🚀 Starting Day 6: LangChain Basics")
    print("Learning: Chains, prompt templates, LangChain patterns")
    
    # Show comparison with Day 1
    compare_with_day1()
    
    try:
        # Create and run the chatbot
        chatbot = LangChainChatbot()
        chatbot.run()
        
    except ImportError as e:
        print(f"\\n❌ Import Error: {e}")
        print("\\n💡 Make sure to install LangChain:")
        print("   pip install langchain langchain-openai")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()