#!/usr/bin/env python3
"""
Day 2 Practice: Experiment with different system prompts and roles.

Try creating different AI assistants with various personalities and expertise.
"""

import os
import sys
import requests
import json
from typing import Optional, List, Dict


class CustomRoleAssistant:
    """A flexible assistant that can take on different roles based on system prompts."""
    
    def __init__(self, role_name: str, system_prompt: str, api_key: Optional[str] = None):
        """Initialize with a custom role and system prompt."""
        self.role_name = role_name
        self.system_prompt = system_prompt
        
        # Use provided key or environment variable
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or "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"
        
        if not self.api_key:
            print("❌ Error: OpenRouter API key not found!")
            sys.exit(1)
            
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": f"Day 2 Practice - {role_name}"
        }
        
    def chat(self, user_input: str) -> str:
        """Simple single-turn chat with the role-based assistant."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"❌ API Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"


def test_different_roles():
    """Test the same question with different AI roles."""
    
    # Define different roles and their system prompts
    roles = {
        "Fitness Coach": """You are an enthusiastic fitness coach with 10 years of experience. 
        You're motivational, energetic, and always focus on practical, achievable fitness goals. 
        Keep responses encouraging and action-oriented. Always ask about current fitness level.""",
        
        "Philosophy Professor": """You are a philosophy professor who loves to explore deep questions. 
        You encourage critical thinking, present multiple perspectives, and often reference famous philosophers. 
        You're thoughtful, intellectual, and help people examine their assumptions.""",
        
        "Startup Mentor": """You are a successful startup founder and mentor. You've built and sold 2 companies. 
        You give practical, no-nonsense advice about entrepreneurship. You focus on execution, customer needs, 
        and avoiding common pitfalls. You're direct but supportive.""",
        
        "Creative Writer": """You are a creative writing teacher who inspires storytelling. 
        You help people explore imagination, develop characters, and craft compelling narratives. 
        You're artistic, encouraging, and always see creative potential in ideas."""
    }
    
    # Test question
    question = "I want to start something new but I'm afraid of failing. What should I do?"
    
    print("🎭 TESTING DIFFERENT AI ROLES")
    print("=" * 50)
    print(f"Question: {question}")
    print("=" * 50)
    
    for role_name, system_prompt in roles.items():
        print(f"\n🎯 {role_name.upper()}")
        print("-" * 30)
        
        assistant = CustomRoleAssistant(role_name, system_prompt)
        response = assistant.chat(question)
        print(response)
        print("-" * 30)


def create_custom_assistant():
    """Let the user create their own custom role assistant."""
    print("\n🛠️  CREATE YOUR OWN AI ASSISTANT")
    print("=" * 40)
    
    role_name = input("What role should your AI play? (e.g., 'Life Coach', 'Tech Tutor'): ").strip()
    if not role_name:
        role_name = "Custom Assistant"
    
    print(f"\nNow describe {role_name}'s personality and expertise:")
    print("Example: 'You are a patient piano teacher who explains music theory simply...'")
    
    system_prompt = input("\nSystem prompt: ").strip()
    if not system_prompt:
        system_prompt = "You are a helpful assistant."
    
    assistant = CustomRoleAssistant(role_name, system_prompt)
    
    print(f"\n🎉 Created {role_name}! Ask them anything:")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input(f"{role_name} > ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input:
            response = assistant.chat(user_input)
            print(f"\n{response}\n")


def main():
    """Practice with different roles and system prompts."""
    print("🚀 Day 2 Practice: Experimenting with Roles & System Prompts")
    print("\nChoose an option:")
    print("1. Test different pre-built roles")
    print("2. Create your own custom assistant")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_different_roles()
    elif choice == "2":
        create_custom_assistant()
    else:
        print("Invalid choice. Running role comparison demo...")
        test_different_roles()


if __name__ == "__main__":
    main()