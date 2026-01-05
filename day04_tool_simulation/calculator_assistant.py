#!/usr/bin/env python3
"""
Day 4 - Tool Simulation: Calculator Assistant

This script demonstrates how an LLM can decide when to use external tools (calculator)
and then Python executes the functions. This is manual tool calling - the foundation
for understanding more advanced agent tool systems.

Key Learning Goals:
- Tool-calling concept (manual)
- LLM decision making about when tools are needed
- Separating reasoning from execution
- Structured tool requests and responses
"""

import json
import os
import sys
import re
import math
import requests
from typing import Dict, Any, List, Optional

class CalculatorTools:
    """Collection of calculator tools that can be called by the LLM."""
    
    @staticmethod
    def calculate(expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression as string (e.g., "2 + 3 * 4")
            
        Returns:
            Dict with result and metadata
        """
        try:
            # Remove any dangerous characters/functions for safety
            safe_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            print("🐍 File: day04_tool_simulation/calculator_assistant.py | Line: 41 | calculate ~ safe_expression",safe_expression)
            
            # Replace common mathematical notations
            safe_expression = safe_expression.replace('×', '*').replace('÷', '/')
            
            # Evaluate the expression
            result = eval(safe_expression, {"__builtins__": {}}, {})
            
            return {
                "tool": "calculate",
                "expression": expression,
                "safe_expression": safe_expression,
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "tool": "calculate",
                "expression": expression,
                "result": None,
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def square_root(number: float) -> Dict[str, Any]:
        """Calculate square root of a number."""
        try:
            result = math.sqrt(number)
            return {
                "tool": "square_root",
                "input": number,
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "tool": "square_root",
                "input": number,
                "result": None,
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def power(base: float, exponent: float) -> Dict[str, Any]:
        """Calculate base raised to exponent."""
        try:
            result = math.pow(base, exponent)
            return {
                "tool": "power",
                "base": base,
                "exponent": exponent,
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "tool": "power",
                "base": base,
                "exponent": exponent,
                "result": None,
                "success": False,
                "error": str(e)
            }

class CalculatorAssistant:
    """AI assistant that can decide when to use calculator tools."""
    
    def __init__(self):
        """Initialize the calculator assistant with OpenRouter API."""
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
            "X-Title": "AI Learning Day 4 - Tool Simulation"
        }
        
        self.tools = CalculatorTools()
        
    def create_tool_decision_prompt(self, user_query: str) -> str:
        """Create a prompt for the LLM to decide if tools are needed."""
        return f"""You are a helpful assistant with access to calculator tools. Your job is to:

1. Analyze the user's query
2. Decide if mathematical calculations are needed
3. If calculations are needed, specify which tool(s) to use

Available Tools:
- calculate(expression): For basic math expressions like "2 + 3 * 4"
- square_root(number): For square root calculations
- power(base, exponent): For exponentiation like "2^8"

User Query: "{user_query}"

You MUST respond with valid JSON in this format:

If NO calculation is needed:
{{
    "needs_calculation": false,
    "response": "your direct answer here"
}}

If calculation IS needed:
{{
    "needs_calculation": true,
    "tools_to_use": [
        {{
            "tool": "calculate",
            "parameters": {{"expression": "2 + 3 * 4"}}
        }}
    ],
    "reasoning": "explanation of why you need these calculations"
}}

Be specific with tool parameters. Only use tools when mathematical computation is actually required."""

    def ask_llm_for_decision(self, user_query: str) -> Dict[str, Any]:
        """Ask the LLM to decide if tools are needed."""
        try:
            prompt = self.create_tool_decision_prompt(user_query)
            
            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            print(f"🔍 Debug: Making API request to {self.api_url}")
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            print(f"🔍 Debug: Response status: {response.status_code}")
            print(f"🔍 Debug: Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"🔍 Debug: Full response text: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            response_text = response.text
            print(f"🔍 Debug: Raw response: {response_text[:200]}...")
            
            if not response_text or response_text.strip() == "":
                return {"success": False, "decision": None, "error": "Empty response from API"}
            
            try:
                response_data = response.json()
                if 'choices' not in response_data or not response_data['choices']:
                    return {"success": False, "decision": None, "error": f"Invalid API response structure: {response_data}"}
                
                result_text = response_data['choices'][0]['message']['content'].strip()
                print(f"🔍 Debug: LLM response content: {result_text}")
                
                # Clean up markdown code blocks if present
                if result_text.startswith('```json'):
                    result_text = result_text.replace('```json', '').replace('```', '').strip()
                elif result_text.startswith('```'):
                    result_text = result_text.replace('```', '').strip()
                
                print(f"🔍 Debug: Cleaned JSON text: {result_text}")
                
                # Parse JSON response
                try:
                    decision = json.loads(result_text)
                    return {"success": True, "decision": decision, "error": None}
                except json.JSONDecodeError as e:
                    return {"success": False, "decision": None, "error": f"JSON parsing error: {e}", "raw_response": result_text}
            except json.JSONDecodeError as e:
                return {"success": False, "decision": None, "error": f"Failed to parse API response as JSON: {e}", "raw_response": response_text}
                
        except Exception as e:
            return {"success": False, "decision": None, "error": str(e)}
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters."""
        if tool_name == "calculate":
            return self.tools.calculate(parameters.get("expression", ""))
        elif tool_name == "square_root":
            return self.tools.square_root(parameters.get("number", 0))
        elif tool_name == "power":
            return self.tools.power(parameters.get("base", 0), parameters.get("exponent", 0))
        else:
            return {
                "tool": tool_name,
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def generate_final_response(self, user_query: str, tool_results: List[Dict[str, Any]]) -> str:
        """Generate final response using tool results."""
        try:
            # Create context with tool results
            tool_context = "\\n".join([
                f"Tool: {result['tool']}, Result: {result.get('result', 'Error: ' + str(result.get('error')))}"
                for result in tool_results
            ])
            
            prompt = f"""Based on the user's query and the tool execution results, provide a helpful response.

User Query: "{user_query}"

Tool Results:
{tool_context}

Provide a natural, conversational response that incorporates the calculation results. Be clear about what calculations were performed."""

            payload = {
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
            
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content'].strip()
            else:
                return f"I calculated the results but had trouble formatting the response. Here are the raw results: {tool_context}"
                
        except Exception as e:
            return f"I calculated the results but had trouble formatting the response: {str(e)}"
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the complete tool simulation pipeline."""
        print("🤔 Analyzing query...")
        
        # Step 1: Ask LLM to decide if tools are needed
        decision_result = self.ask_llm_for_decision(user_query)
        
        if not decision_result["success"]:
            return {
                "success": False,
                "response": f"Error getting decision: {decision_result['error']}",
                "details": decision_result
            }
        
        decision = decision_result["decision"]
        
        # Step 2: Handle based on decision
        if not decision.get("needs_calculation", False):
            return {
                "success": True,
                "response": decision.get("response", "No calculation needed."),
                "used_tools": False,
                "details": {"decision": decision}
            }
        
        # Step 3: Execute tools if needed
        print("🔧 Tools needed! Executing calculations...")
        tool_results = []
        
        for tool_request in decision.get("tools_to_use", []):
            tool_name = tool_request.get("tool")
            parameters = tool_request.get("parameters", {})
            
            print(f"   Executing: {tool_name}({parameters})")
            result = self.execute_tool(tool_name, parameters)
            tool_results.append(result)
        
        # Step 4: Generate final response
        print("💬 Generating response...")
        final_response = self.generate_final_response(user_query, tool_results)
        
        return {
            "success": True,
            "response": final_response,
            "used_tools": True,
            "tool_results": tool_results,
            "details": {"decision": decision, "reasoning": decision.get("reasoning")}
        }

def main():
    """Main interactive loop for the calculator assistant."""
    print("🔢 Calculator Assistant - Day 4: Tool Simulation")
    print("=" * 55)
    print("Ask me anything! I'll decide if I need to use calculator tools.")
    print("Examples:")
    print("  - 'What is 25 * 8 + 12?'")
    print("  - 'What's the square root of 144?'") 
    print("  - 'Calculate 2 to the power of 10'")
    print("  - 'What is the capital of France?' (no tools needed)")
    print("Type 'quit' to exit.\\n")
    
    assistant = CalculatorAssistant()
    
    # Demo examples
    examples = [
        "What is 15 * 8 + 32?",
        "What's the square root of 64?",
        "What is the capital of France?"
    ]
    
    print("🔍 Here are some example interactions:")
    for example in examples:
        print(f"\\n👤 User: {example}")
        result = assistant.process_query(example)
        print(f"🤖 Assistant: {result['response']}")
        
        if result.get('used_tools'):
            print("🔧 Tools used:")
            for tool_result in result.get('tool_results', []):
                if tool_result['success']:
                    print(f"   ✅ {tool_result['tool']}: {tool_result['result']}")
                else:
                    print(f"   ❌ {tool_result['tool']}: {tool_result['error']}")
    
    print("\\n" + "=" * 55)
    print("Now try your own questions:")
    
    while True:
        try:
            user_input = input("\\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Keep calculating!")
                break
            
            if not user_input:
                print("Please ask a question.")
                continue
            
            result = assistant.process_query(user_input)
            print(f"🤖 Assistant: {result['response']}")
            
            # Show tool usage details
            if result.get('used_tools'):
                print("\\n🔍 Behind the scenes:")
                print(f"   Reasoning: {result['details'].get('reasoning', 'N/A')}")
                print("   Tools executed:")
                for tool_result in result.get('tool_results', []):
                    if tool_result['success']:
                        print(f"     ✅ {tool_result['tool']}: {tool_result.get('result')}")
                    else:
                        print(f"     ❌ {tool_result['tool']}: {tool_result.get('error')}")
            else:
                print("   💭 No calculations needed - answered directly!")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye! Keep calculating!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()