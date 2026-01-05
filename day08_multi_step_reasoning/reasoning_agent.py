#!/usr/bin/env python3
"""
Day 8 - Multi-Step Reasoning: Think-Plan-Act Agent

This script demonstrates multi-step reasoning by building an agent that can:
1. THINK: Analyze complex problems
2. PLAN: Break them into steps  
3. ACT: Execute coordinated actions
4. REFLECT: Review and continue if needed

Key Learning Goals:
- Reasoning loops and iterative problem solving
- Agent scratchpad for maintaining state
- Multi-step tool coordination
- Complex problem decomposition
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import re

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class CalculatorTool(BaseTool):
    """Enhanced calculator tool for multi-step reasoning."""
    
    name: str = "calculator"
    description: str = "Performs mathematical calculations. Input: mathematical expression like '2 + 2' or 'pi * 5^2'."
    
    def _run(self, expression: str) -> str:
        """Execute calculation with enhanced math support."""
        try:
            # Clean and prepare expression
            expression = expression.replace("π", "3.14159").replace("pi", "3.14159")
            expression = expression.replace("^", "**")  # Python exponent operator
            expression = re.sub(r'[^0-9+\-*/().\s**]', '', expression)
            
            # Safe evaluation
            result = eval(expression)
            return f"{expression} = {result}"
            
        except Exception as e:
            return f"Calculation error for '{expression}': {str(e)}"

class DateTimeTool(BaseTool):
    """Enhanced datetime tool with more capabilities."""
    
    name: str = "datetime"
    description: str = "Gets current date/time information, calculates time differences, formats dates."
    
    def _run(self, query: str = "") -> str:
        """Execute datetime operations."""
        try:
            now = datetime.now()
            
            # Basic time info
            basic_info = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"), 
                "day": now.strftime("%A"),
                "formatted": now.strftime("%B %d, %Y at %I:%M %p")
            }
            
            return f"Current: {basic_info['formatted']} ({basic_info['day']})"
            
        except Exception as e:
            return f"DateTime error: {str(e)}"

class ReasoningStep:
    """Represents a single step in the reasoning process."""
    
    def __init__(self, step_type: str, content: str, result: str = ""):
        self.step_type = step_type  # THINK, PLAN, ACT, REFLECT
        self.content = content
        self.result = result
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"{self.step_type}: {self.content}"

class AgentScratchpad:
    """Maintains the agent's reasoning state across multiple steps."""
    
    def __init__(self):
        self.steps: List[ReasoningStep] = []
        self.current_plan: List[str] = []
        self.completed_actions: List[str] = []
        self.final_answer: str = ""
    
    def add_step(self, step_type: str, content: str, result: str = ""):
        """Add a reasoning step to the scratchpad."""
        step = ReasoningStep(step_type, content, result)
        self.steps.append(step)
    
    def get_context(self) -> str:
        """Get current reasoning context for the LLM."""
        context = "=== AGENT SCRATCHPAD ===\\n"
        for step in self.steps:
            context += f"{step}\\n"
        
        if self.current_plan:
            context += "\\nCURRENT PLAN:\\n"
            for i, plan_item in enumerate(self.current_plan, 1):
                status = "✓" if plan_item in self.completed_actions else "○"
                context += f"{status} {i}. {plan_item}\\n"
        
        return context
    
    def is_task_complete(self) -> bool:
        """Check if the reasoning task appears complete."""
        return len(self.final_answer) > 0 and len(self.completed_actions) >= len(self.current_plan)

class MultiStepReasoningAgent:
    """An agent that uses multi-step reasoning to solve complex problems."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the multi-step reasoning agent."""
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
            temperature=0.1,  # Low temperature for consistent reasoning
            max_tokens=800
        )
        
        # Available tools
        self.tools = {
            "calculator": CalculatorTool(),
            "datetime": DateTimeTool()
        }
        
        # Create reasoning prompt template
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a multi-step reasoning agent. When given a complex problem, you should:

1. THINK: Analyze the problem and understand what needs to be done
2. PLAN: Break it down into clear, actionable steps  
3. ACT: Execute steps using available tools when needed
4. REFLECT: Review progress and determine next actions

Available tools:
- calculator: For any mathematical calculations
- datetime: For current date/time information

ALWAYS follow this format for your response:

🤔 THINK:
[Your analysis of the problem]

📋 PLAN:
[Numbered list of steps to solve the problem]

⚡ ACT:
[If you need to use a tool, format exactly as:]
TOOL: tool_name
INPUT: tool_input

[Otherwise, describe what you're doing]

🔄 REFLECT:
[Review what you've accomplished and what's next]

Continue until the problem is fully solved."""),
            ("human", """Problem: {problem}

{context}

Continue with the next reasoning step:""")
        ])
        
        # Create reasoning chain
        self.reasoning_chain = self.reasoning_prompt | self.llm
        
        print("🧠 Multi-step reasoning agent initialized!")
        print(f"🔧 Available tools: {list(self.tools.keys())}")
    
    def reason_step(self, problem: str, scratchpad: AgentScratchpad) -> str:
        """Execute one step of reasoning."""
        try:
            # Get current context
            context = scratchpad.get_context()
            
            # Get reasoning response
            response = self.reasoning_chain.invoke({
                "problem": problem,
                "context": context
            })
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
            
        except Exception as e:
            return f"❌ Reasoning error: {str(e)}"
    
    def parse_reasoning_response(self, response: str, scratchpad: AgentScratchpad):
        """Parse the reasoning response and update scratchpad."""
        sections = {
            "THINK": "",
            "PLAN": "",
            "ACT": "",
            "REFLECT": ""
        }
        
        current_section = None
        lines = response.split('\\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('🤔 THINK:'):
                current_section = "THINK"
                sections[current_section] = line.replace('🤔 THINK:', '').strip()
            elif line.startswith('📋 PLAN:'):
                current_section = "PLAN"
                sections[current_section] = line.replace('📋 PLAN:', '').strip()
            elif line.startswith('⚡ ACT:'):
                current_section = "ACT"
                sections[current_section] = line.replace('⚡ ACT:', '').strip()
            elif line.startswith('🔄 REFLECT:'):
                current_section = "REFLECT"
                sections[current_section] = line.replace('🔄 REFLECT:', '').strip()
            elif current_section and line:
                sections[current_section] += " " + line
        
        # Add steps to scratchpad
        for section_type, content in sections.items():
            if content.strip():
                scratchpad.add_step(section_type, content.strip())
        
        # Handle tool usage in ACT section
        if "TOOL:" in sections["ACT"]:
            self.handle_tool_usage(sections["ACT"], scratchpad)
    
    def handle_tool_usage(self, act_section: str, scratchpad: AgentScratchpad):
        """Handle tool usage from ACT section."""
        lines = act_section.split('\\n')
        tool_name = None
        tool_input = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("INPUT:"):
                tool_input = line.replace("INPUT:", "").strip()
        
        if tool_name and tool_input and tool_name in self.tools:
            print(f"🔧 Executing {tool_name} with input: '{tool_input}'")
            result = self.tools[tool_name]._run(tool_input)
            scratchpad.add_step("TOOL_RESULT", f"{tool_name}: {result}")
            scratchpad.completed_actions.append(f"Used {tool_name}: {tool_input}")
            return result
        
        return None
    
    def solve_problem(self, problem: str, max_steps: int = 10) -> str:
        """Solve a complex problem using multi-step reasoning."""
        print(f"\\n🧠 Starting multi-step reasoning for: {problem}")
        print("=" * 60)
        
        scratchpad = AgentScratchpad()
        
        for step in range(max_steps):
            print(f"\\n--- Reasoning Step {step + 1} ---")
            
            # Get next reasoning step
            response = self.reason_step(problem, scratchpad)
            
            # Parse and handle the response
            self.parse_reasoning_response(response, scratchpad)
            
            # Show current thinking
            if scratchpad.steps:
                latest_step = scratchpad.steps[-1]
                print(f"{latest_step.step_type}: {latest_step.content}")
                if latest_step.result:
                    print(f"Result: {latest_step.result}")
            
            # Check if task appears complete
            if "final answer" in response.lower() or "complete" in response.lower():
                break
            
            # Prevent infinite loops
            if step >= max_steps - 1:
                print("\\n⚠️ Maximum reasoning steps reached")
                break
        
        # Generate final summary
        return self.generate_final_answer(scratchpad)
    
    def generate_final_answer(self, scratchpad: AgentScratchpad) -> str:
        """Generate a final answer based on the reasoning process."""
        context = scratchpad.get_context()
        
        final_prompt = f"""Based on this reasoning process, provide a clear, comprehensive final answer:

{context}

Final Answer:"""
        
        try:
            final_response = self.llm.invoke([HumanMessage(content=final_prompt)])
            return final_response.content if hasattr(final_response, 'content') else str(final_response)
        except Exception as e:
            return f"Error generating final answer: {str(e)}"
    
    def run(self):
        """Main interaction loop."""
        print("\\n🧠 Multi-Step Reasoning Agent - Day 8")
        print("=" * 55)
        print("This agent can break down complex problems into steps!")
        print("\\nTry asking:")
        print("  • 'Calculate the area of a circle with radius 5, then find what percentage that is of a square with side 10'")
        print("  • 'If I save $50 per month starting now, how much will I have in 2 years? What's that as a daily amount?'")
        print("  • 'What time is it now, and how many hours until midnight?'")
        print("\\nType 'quit' to exit")
        print("-" * 55)
        
        while True:
            try:
                # Get user input
                user_input = input("\\n👤 You: ").strip()
                
                # Handle exit
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\\n🧠 Thanks for exploring multi-step reasoning! 🚀")
                    break
                
                # Skip empty input
                if not user_input:
                    print("Please enter a problem or type 'quit' to exit.")
                    continue
                
                # Solve the problem using multi-step reasoning
                answer = self.solve_problem(user_input)
                
                print(f"\\n🎯 Final Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye! Thanks for exploring multi-step reasoning!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")

def compare_with_day7():
    """Show comparison between Day 7 (single tool) and Day 8 (multi-step) approaches."""
    print("\\n📊 Day 7 vs Day 8 Reasoning Comparison")
    print("=" * 55)
    print("\\n**Day 7 (Direct Tool Calling)**:")
    print("""
    User: "What's 15 * 23?"
    Agent: Uses calculator → Returns result
    
    Single step, direct response
    """)
    
    print("\\n**Day 8 (Multi-Step Reasoning)**:")
    print("""
    User: "Calculate circle area (r=5), then find percentage of square (side=10)"
    
    🤔 THINK: Need circle area, square area, then percentage
    📋 PLAN: 1. Circle area  2. Square area  3. Percentage
    ⚡ ACT: calculator(π*5²) → calculator(10²) → calculator(ratio)
    🔄 REFLECT: All steps complete, provide final answer
    
    Multi-step, coordinated reasoning
    """)
    
    print("\\n✅ **Multi-Step Reasoning Benefits**:")
    print("  • Complex problem decomposition")
    print("  • Coordinated tool usage")
    print("  • Reasoning transparency")
    print("  • Iterative problem solving")
    print("  • Maintains context across steps")

def main():
    """Main function."""
    print("🚀 Starting Day 8: Multi-Step Reasoning")
    print("Learning: Think-Plan-Act patterns and reasoning loops")
    
    # Show comparison with Day 7
    compare_with_day7()
    
    try:
        # Create and run the reasoning agent
        agent = MultiStepReasoningAgent()
        agent.run()
        
    except ImportError as e:
        print(f"\\n❌ Import Error: {e}")
        print("\\n💡 Make sure to install required packages:")
        print("   pip install langchain langchain-openai")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()