#!/usr/bin/env python3
"""
Day 3 - Structured Output: Expense Categorizer

This script takes expense descriptions and categorizes them using an LLM,
returning structured JSON output with category and confidence score.

Key Learning Goals:
- Structured output from LLMs
- JSON validation and parsing
- Prompt engineering for consistent formatting
"""

import json
import os
import sys
import requests
from typing import Dict, Any

# Categories for expense classification
EXPENSE_CATEGORIES = [
    "food_dining",
    "transportation", 
    "entertainment",
    "utilities",
    "healthcare",
    "shopping_retail",
    "groceries",
    "rent_housing",
    "travel",
    "education",
    "insurance",
    "other"
]

class ExpenseCategorizer:
    def __init__(self):
        """Initialize the expense categorizer with OpenRouter API."""
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
            "X-Title": "AI Learning Day 3 - Expense Categorizer"
        }
    
    def create_categorization_prompt(self, expense_description: str) -> str:
        """Create a structured prompt for expense categorization."""
        categories_list = ", ".join(EXPENSE_CATEGORIES)
        
        return f"""You are an expense categorization assistant. Analyze the expense description and categorize it.

Expense Description: "{expense_description}"

Available Categories: {categories_list}

You MUST respond with valid JSON in exactly this format:
{{
    "category": "category_name",
    "confidence": confidence_score,
    "reasoning": "brief explanation"
}}

Rules:
- category: must be one of the available categories
- confidence: float between 0.0 and 1.0 (1.0 = very confident, 0.5 = uncertain)
- reasoning: brief 1-sentence explanation for the categorization
- respond with ONLY the JSON object, no additional text

Example:
{{
    "category": "food_dining",
    "confidence": 0.9,
    "reasoning": "Restaurant name and typical dining expense amount"
}}"""

    def categorize_expense(self, description: str) -> Dict[str, Any]:
        """
        Categorize a single expense description.
        
        Args:
            description: The expense description to categorize
            
        Returns:
            Dict containing category, confidence, and reasoning
        """
        try:
            prompt = self.create_categorization_prompt(description)
            
            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent output
                "max_tokens": 150
            }
            
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            response_data = response.json()
            result_text = response_data['choices'][0]['message']['content'].strip()
            
            # Parse the JSON response
            try:
                result = json.loads(result_text)
                
                # Validate the structure
                if not all(key in result for key in ["category", "confidence", "reasoning"]):
                    raise ValueError("Missing required keys in response")
                
                if result["category"] not in EXPENSE_CATEGORIES:
                    raise ValueError(f"Invalid category: {result['category']}")
                
                if not (0.0 <= result["confidence"] <= 1.0):
                    raise ValueError(f"Invalid confidence score: {result['confidence']}")
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Raw response: {result_text}")
                return {
                    "category": "other",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response as JSON"
                }
                
        except Exception as e:
            print(f"❌ Error categorizing expense: {e}")
            return {
                "category": "other",
                "confidence": 0.0,
                "reasoning": f"Error during categorization: {str(e)}"
            }

    def format_result(self, description: str, result: Dict[str, Any]) -> str:
        """Format the categorization result for display."""
        confidence_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
        
        return f"""
📝 Expense: "{description}"
📂 Category: {result["category"]}
📊 Confidence: {result["confidence"]:.1f} [{confidence_bar}]
💭 Reasoning: {result["reasoning"]}
"""

def main():
    """Main interactive loop for the expense categorizer."""
    print("💰 Expense Categorizer - Day 3: Structured Output")
    print("=" * 50)
    print("Enter expense descriptions to categorize them automatically.")
    print("Examples: 'Starbucks coffee $5.50', 'Uber ride downtown', 'Netflix subscription'")
    print("Type 'quit' to exit.\n")
    
    categorizer = ExpenseCategorizer()
    
    # Example expenses for demonstration
    examples = [
        "Starbucks coffee and pastry $8.75",
        "Uber ride to airport $45.20", 
        "Netflix monthly subscription $15.99",
        "Grocery shopping at Whole Foods $127.43",
        "Dinner at Italian restaurant $89.50"
    ]
    
    print("🔍 Here are some example categorizations:")
    for example in examples[:3]:  # Show first 3 examples
        result = categorizer.categorize_expense(example)
        print(categorizer.format_result(example, result))
    
    print("\n" + "=" * 50)
    print("Now try your own expenses:")
    
    while True:
        try:
            user_input = input("\n💸 Enter expense description: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Keep tracking those expenses!")
                break
            
            if not user_input:
                print("Please enter an expense description.")
                continue
            
            print("🤔 Analyzing...")
            result = categorizer.categorize_expense(user_input)
            print(categorizer.format_result(user_input, result))
            
            # Show raw JSON for learning purposes
            print("🔧 Raw JSON output:")
            print(json.dumps(result, indent=2))
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Keep tracking those expenses!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()