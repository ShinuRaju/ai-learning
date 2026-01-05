#!/usr/bin/env python3
"""
Day 10 - Mini Use Case: Expense Explainer Agent

This script demonstrates a practical AI application by building an expense analyzer that:
1. Takes a list of expenses as input
2. Categorizes and analyzes spending patterns
3. Provides insights and recommendations
4. Uses memory to track conversation context
5. Combines tools, reasoning, and structured output

Key Learning Goals:
- Practical AI application development
- Combining multiple AI techniques into one system
- Real-world use case implementation
- Data analysis with LLM reasoning
- Actionable insights generation
"""

import os
import sys
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
import json
import re

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

class ExpenseCategorizerTool(BaseTool):
    """Tool to categorize expenses into standard categories."""
    
    name: str = "expense_categorizer"
    description: str = "Categorizes expenses into standard categories like Food, Transportation, Entertainment, etc."
    
    def _run(self, expenses_text: str) -> str:
        """Categorize a list of expenses."""
        try:
            # Standard expense categories
            categories = {
                "Food & Dining": ["restaurant", "food", "coffee", "lunch", "dinner", "grocery", "starbucks", "pizza"],
                "Transportation": ["uber", "lyft", "gas", "taxi", "parking", "metro", "bus", "train"],
                "Entertainment": ["movie", "netflix", "spotify", "concert", "game", "bar", "club"],
                "Shopping": ["amazon", "target", "walmart", "clothes", "shoes", "electronics"],
                "Bills & Utilities": ["electric", "gas", "water", "phone", "internet", "rent", "insurance"],
                "Health": ["doctor", "pharmacy", "hospital", "medicine", "gym", "fitness"],
                "Travel": ["hotel", "flight", "airbnb", "vacation", "booking"],
                "Other": []
            }
            
            lines = expenses_text.strip().split('\n')
            categorized = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                # Parse line (assuming format: "amount description" or "description amount")
                parts = line.strip().split()
                amount = None
                description = ""
                
                # Find amount (starts with $ or is a number)
                for part in parts:
                    if part.startswith('$') or part.replace('.', '').replace('-', '').isdigit():
                        amount = part.replace('$', '')
                        break
                
                # Get description (everything else)
                description = ' '.join([p for p in parts if p != f"${amount}" and p != amount])
                
                # Categorize based on keywords
                category = "Other"
                desc_lower = description.lower()
                
                for cat, keywords in categories.items():
                    if any(keyword in desc_lower for keyword in keywords):
                        category = cat
                        break
                
                categorized.append({
                    "amount": amount or "0",
                    "description": description,
                    "category": category,
                    "original": line.strip()
                })
            
            return json.dumps(categorized, indent=2)
            
        except Exception as e:
            return f"Categorization error: {str(e)}"

class ExpenseCalculatorTool(BaseTool):
    """Tool to calculate expense totals and statistics."""
    
    name: str = "expense_calculator"
    description: str = "Calculates total expenses, averages, and category breakdowns from expense data."
    
    def _run(self, categorized_data: str) -> str:
        """Calculate expense statistics."""
        try:
            expenses = json.loads(categorized_data)
            
            total = 0
            category_totals = {}
            valid_expenses = []
            
            for expense in expenses:
                try:
                    amount_str = expense['amount'].replace('$', '').replace(',', '')
                    amount = float(amount_str)
                    category = expense['category']
                    
                    total += amount
                    valid_expenses.append(amount)
                    
                    if category not in category_totals:
                        category_totals[category] = 0
                    category_totals[category] += amount
                    
                except (ValueError, KeyError):
                    continue
            
            # Calculate statistics
            stats = {
                "total_expenses": round(total, 2),
                "expense_count": len(valid_expenses),
                "average_expense": round(total / len(valid_expenses), 2) if valid_expenses else 0,
                "category_totals": {cat: round(amount, 2) for cat, amount in category_totals.items()},
                "top_category": max(category_totals, key=category_totals.get) if category_totals else "None"
            }
            
            return json.dumps(stats, indent=2)
            
        except Exception as e:
            return f"Calculation error: {str(e)}"

class ExpenseExplainerAgent:
    """An agent that analyzes expenses and provides insights."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the expense explainer agent."""
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
            temperature=0.3,  # Lower for more consistent analysis
            max_tokens=800
        )
        
        # Initialize tools
        self.tools = {
            "categorizer": ExpenseCategorizerTool(),
            "calculator": ExpenseCalculatorTool()
        }
        
        # Initialize conversation memory
        self.conversation_history = []
        
        # Create analysis prompt
        self.setup_analysis_chain()
        
        print("💰 Expense Explainer Agent initialized!")
        print("📊 I can analyze your expenses and provide insights.")
    
    def setup_analysis_chain(self):
        """Set up the expense analysis chain."""
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial advisor and expense analyst. Your job is to:

1. Analyze expense data provided by tools
2. Identify spending patterns and trends
3. Provide actionable insights and recommendations
4. Give personalized financial advice
5. Be encouraging but honest about spending habits

When analyzing expenses:
- Focus on the biggest categories first
- Look for unusual or concerning patterns  
- Suggest realistic improvements
- Consider the user's context and previous conversations
- Provide specific, actionable advice

Format your response with clear sections:
📊 EXPENSE SUMMARY
💡 KEY INSIGHTS  
⚠️ AREAS OF CONCERN
✅ RECOMMENDATIONS
💪 ACTION ITEMS

Be conversational and helpful, not judgmental."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Please analyze these expenses and provide insights:

CATEGORIZED EXPENSES:
{categorized_data}

CALCULATED STATISTICS:
{statistics}

ORIGINAL INPUT:
{original_input}

Provide a comprehensive analysis with actionable recommendations.""")
        ])
        
        self.analysis_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.conversation_history
            )
            | self.analysis_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def analyze_expenses(self, expenses_input: str) -> str:
        """Analyze expenses and provide insights."""
        try:
            print("🔄 Processing expenses...")
            
            # Step 1: Categorize expenses
            print("📋 Categorizing expenses...")
            categorized_data = self.tools["categorizer"]._run(expenses_input)
            
            # Step 2: Calculate statistics
            print("🧮 Calculating statistics...")
            statistics = self.tools["calculator"]._run(categorized_data)
            
            # Step 3: Generate insights
            print("💡 Generating insights...")
            analysis = self.analysis_chain.invoke({
                "categorized_data": categorized_data,
                "statistics": statistics,
                "original_input": expenses_input
            })
            
            # Save to conversation history
            self.conversation_history.append(HumanMessage(content=f"Expenses to analyze:\n{expenses_input}"))
            self.conversation_history.append(AIMessage(content=analysis))
            
            return analysis
            
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    def get_sample_expenses(self) -> str:
        """Get sample expense data for testing."""
        return """$45.67 Whole Foods grocery shopping
$12.50 Uber ride downtown
$8.99 Netflix subscription
$150.00 Electric bill
$25.00 Starbucks coffee and breakfast
$89.99 Amazon electronics purchase
$35.00 Gas station fill-up
$75.00 Dinner at Italian restaurant
$15.99 Spotify premium
$200.00 Monthly gym membership
$65.00 Doctor visit copay
$120.00 Groceries at Target
$18.00 Movie tickets
$95.00 Phone bill
$250.00 Hotel booking for weekend trip"""
    
    def chat_about_expenses(self, user_input: str) -> str:
        """Have a conversation about expenses and financial advice."""
        try:
            # Simple chat prompt for follow-up questions
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial advisor helping someone understand their expenses. 
                You have access to their previous expense analysis and can answer follow-up questions.
                Be helpful, encouraging, and provide specific actionable advice.
                
                If they ask about new expenses, suggest they provide a list for analysis.
                If they want to compare expenses, help them understand the differences."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            chat_chain = (
                RunnablePassthrough.assign(
                    chat_history=lambda x: self.conversation_history
                )
                | chat_prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = chat_chain.invoke({"input": user_input})
            
            # Save to conversation history
            self.conversation_history.append(HumanMessage(content=user_input))
            self.conversation_history.append(AIMessage(content=response))
            
            return response
            
        except Exception as e:
            return f"❌ Chat error: {str(e)}"
    
    def show_conversation_summary(self):
        """Show conversation history summary."""
        print("\n📋 Conversation Summary:")
        print(f"Total exchanges: {len(self.conversation_history) // 2}")
        
        if self.conversation_history:
            print("\n💬 Recent conversation:")
            recent = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history
            for msg in recent:
                if isinstance(msg, HumanMessage):
                    preview = msg.content[:80].replace('\n', ' ')
                    print(f"  👤 You: {preview}{'...' if len(msg.content) > 80 else ''}")
                elif isinstance(msg, AIMessage):
                    preview = msg.content[:80].replace('\n', ' ')
                    print(f"  💰 Agent: {preview}{'...' if len(msg.content) > 80 else ''}")
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared!")
    
    def run(self):
        """Main interaction loop."""
        print("\n💰 Expense Explainer Agent - Day 10")
        print("=" * 55)
        print("I can analyze your expenses and provide financial insights!")
        print("\nWhat you can do:")
        print("  • Paste expense data for analysis")
        print("  • Type 'sample' for sample expense data")
        print("  • Ask follow-up questions about your spending")
        print("  • Type 'summary' to see conversation history")
        print("  • Type 'clear' to clear conversation")
        print("  • Type 'quit' to exit")
        print("\nExpense format examples:")
        print("  $45.67 Grocery shopping at Whole Foods")
        print("  Uber ride $12.50")
        print("  Netflix subscription $8.99")
        print("-" * 55)
        
        while True:
            try:
                user_input = input("\n💰 Enter expenses or ask a question: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n💰 Thanks for using Expense Explainer! Save money! 💪")
                    break
                
                elif user_input.lower() == 'sample':
                    sample_data = self.get_sample_expenses()
                    print("\n📋 Analyzing sample expense data...")
                    print(f"\n{sample_data}\n")
                    analysis = self.analyze_expenses(sample_data)
                    print(f"\n📊 EXPENSE ANALYSIS:\n{analysis}")
                    continue
                
                elif user_input.lower() == 'summary':
                    self.show_conversation_summary()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                    continue
                
                # Skip empty input
                if not user_input:
                    print("Please enter some expenses or ask a question!")
                    continue
                
                # Detect if input looks like expense data (contains $ or numbers)
                if ('$' in user_input or 
                    any(char.isdigit() for char in user_input) and 
                    len(user_input.split('\n')) > 1):
                    # Treat as expense analysis
                    analysis = self.analyze_expenses(user_input)
                    print(f"\n📊 EXPENSE ANALYSIS:\n{analysis}")
                else:
                    # Treat as conversation/question
                    response = self.chat_about_expenses(user_input)
                    print(f"\n💰 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Take care of your finances!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

def demonstrate_features():
    """Demonstrate key features of the expense explainer."""
    print("\n🎯 Expense Explainer Features Demo")
    print("=" * 40)
    
    print("\n**1. Expense Categorization**")
    print("   Automatically sorts expenses into categories:")
    print("   • Food & Dining, Transportation, Entertainment")
    print("   • Shopping, Bills, Health, Travel, Other")
    
    print("\n**2. Financial Analysis**")
    print("   Calculates key metrics:")
    print("   • Total spending, average expense")
    print("   • Category breakdowns, top spending areas")
    
    print("\n**3. Actionable Insights**")
    print("   Provides personalized advice:")
    print("   • Spending pattern analysis")
    print("   • Budget recommendations")
    print("   • Money-saving suggestions")
    
    print("\n**4. Conversational Interface**")
    print("   Follow-up capabilities:")
    print("   • Ask questions about analysis")
    print("   • Get specific category advice")
    print("   • Compare different expense periods")

def main():
    """Main function."""
    print("🚀 Starting Day 10: Mini Use Case - Expense Explainer")
    print("Learning: Practical AI application combining all previous concepts")
    
    # Show features demo
    demonstrate_features()
    
    try:
        # Create and run the expense explainer agent
        agent = ExpenseExplainerAgent()
        agent.run()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Make sure to install required packages:")
        print("   pip install langchain langchain-openai")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()