#!/usr/bin/env python3
"""
Day 1 - Hello LLM: CLI AI Assistant
A simple command-line AI assistant that demonstrates LLM API basics.

Usage:
    python cli_assistant.py

Learning objectives:
- LLM API basics
- Request/response flow
- Simple CLI interaction
"""

import os
import sys
import requests
import json
from typing import Optional


class CLIAssistant:
    """A simple CLI AI assistant using OpenRouter's API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the assistant with OpenRouter API key."""
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
            "HTTP-Referer": "http://localhost",  # Optional
            "X-Title": "Day 1 CLI Assistant"  # Optional
        }
        
    def get_llm_response(self, user_input: str) -> str:
        """Send user input to LLM and return the response."""
        try:
            # Make API request to OpenRouter
            payload = {
                "model": "mistralai/devstral-2512:free",  # Using free Mistral model
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful AI assistant. Be concise and helpful."
                    },
                    {
                        "role": "user", 
                        "content": user_input
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            # print(json.dumps(result, indent=2))  # Debug: print full API response
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"
        except KeyError as e:
            return f"❌ Unexpected API response format: {str(e)}"
        except Exception as e:
            return f"❌ Error calling LLM API: {str(e)}"
    
    def run(self):
        """Main CLI loop."""
        print("🤖 Hello! I'm your CLI AI Assistant")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Assistant: Goodbye! 👋")
                    break
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                print("\n🤖 Assistant: ", end="")
                
                # Get and display LLM response
                response = self.get_llm_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Assistant: Goodbye! 👋")
                break
            except EOFError:
                print("\n\n🤖 Assistant: Goodbye! 👋")
                break


def main():
    """Main entry point."""
    print("Starting Day 1 - Hello LLM CLI Assistant")
    
    # Create and run the assistant
    assistant = CLIAssistant()
    assistant.run()


if __name__ == "__main__":
    main()