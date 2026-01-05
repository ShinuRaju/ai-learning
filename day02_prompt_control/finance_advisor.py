#!/usr/bin/env python3
"""
Day 2 - Prompt Control: Role-based Finance Advisor
A CLI assistant that acts as a finance advisor using system prompts.

Usage:
    python finance_advisor.py

Learning objectives:
- System vs user prompts
- Prompt discipline
- Role-based AI behavior
"""

import os
import sys
import requests
import json
from typing import Optional, List, Dict


class FinanceAdvisor:
    """A role-based AI assistant that acts as a finance advisor."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the finance advisor with OpenRouter API key."""
        # Use provided key or environment variable
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or "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"
        
        if not self.api_key:
            print("❌ Error: OpenRouter API key not found!")
            print("Please set OPENROUTER_API_KEY environment variable or pass it directly.")
            sys.exit(1)
            
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Day 2 Finance Advisor"
        }
        
        # Define the system prompt for our finance advisor role
        self.system_prompt = """You are a professional finance advisor with 15+ years of experience. Your role is to:

1. Provide practical, actionable financial advice
2. Ask clarifying questions when needed
3. Explain complex concepts in simple terms
4. Always consider risk management
5. Be encouraging but realistic
6. Never recommend specific stocks or get-rich-quick schemes

Guidelines:
- Keep responses concise but thorough
- Use examples when helpful
- Always mention relevant risks
- Ask about the user's financial situation when making recommendations
- Focus on fundamentals: budgeting, saving, investing basics

Remember: You're here to educate and guide, not to make decisions for the user."""

        self.conversation_history: List[Dict[str, str]] = []
        
    def get_llm_response(self, user_input: str) -> str:
        """Send user input to LLM with system prompt and return the response."""
        try:
            # Build messages array with system prompt first, then conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_input})
            
            # Make API request
            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": messages,
                "temperature": 0.7,  # Slightly creative but focused
                "max_tokens": 500    # Keep responses concise
            }
            
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']['content']
                
                # Store this exchange in conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Keep only last 6 messages (3 exchanges) to manage context
                if len(self.conversation_history) > 6:
                    self.conversation_history = self.conversation_history[-6:]
                
                return assistant_message
            else:
                return f"❌ API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ Network Error: {str(e)}"
        except json.JSONDecodeError as e:
            return f"❌ JSON Error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected Error: {str(e)}"

    def display_welcome(self):
        """Display welcome message and advisor introduction."""
        print("=" * 60)
        print("💰 PERSONAL FINANCE ADVISOR")
        print("=" * 60)
        print()
        print("👋 Hello! I'm your AI finance advisor.")
        print("I'm here to help you with:")
        print("  • Budgeting and saving strategies")
        print("  • Investment basics and planning")
        print("  • Debt management")
        print("  • Retirement planning")
        print("  • Financial goal setting")
        print()
        print("💡 Tip: The more details you provide about your situation,")
        print("   the better advice I can give you!")
        print()
        print("Type 'quit' or 'exit' to end the session.")
        print("-" * 60)

    def run(self):
        """Run the interactive finance advisor session."""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 What financial question can I help you with? ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n💝 Thank you for using the Finance Advisor!")
                    print("Remember: Start small, be consistent, and invest in your future!")
                    break
                
                # Skip empty input
                if not user_input:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                
                # Get and display AI response
                print("\n💡 Finance Advisor:")
                print("-" * 40)
                
                response = self.get_llm_response(user_input)
                print(response)
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended. Take care of your finances!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'quit' to exit.")


def main():
    """Main function to run the finance advisor."""
    print("🚀 Starting Day 2: Prompt Control - Finance Advisor")
    print("Learning: System prompts, role-based behavior, prompt discipline")
    print()
    
    # Create and run the finance advisor
    advisor = FinanceAdvisor()
    advisor.run()


if __name__ == "__main__":
    main()