#!/usr/bin/env python3
"""
Day 7 - Tool Calling with LangChain: Tool-Enabled Agent

This script demonstrates LangChain's tool calling capabilities by building
an agent that can use multiple tools to answer questions.

Key Learning Goals:
- LangChain Tool abstraction  
- Agents that can decide which tools to use
- Multiple tool coordination
- Tool calling vs manual tool simulation (Day 4)
"""

import os
import sys
from datetime import datetime
from typing import Optional
import re

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class CalculatorTool(BaseTool):
    """A tool for performing mathematical calculations."""
    
    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations. Input should be a mathematical expression like '2 + 2' or '15 * 23'."
    
    def _run(self, expression: str) -> str:
        """Execute the calculator tool."""
        try:
            # Clean the expression (basic security)
            expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Evaluate safely
            result = eval(expression)
            return f"The result of {expression} is {result}"
            
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

class DateTimeTool(BaseTool):
    """A tool for getting current date and time information."""
    
    name: str = "datetime"
    description: str = "Useful for getting current date, time, day of week, or any time-related information."
    
    def _run(self, query: str = "") -> str:
        """Execute the datetime tool."""
        try:
            now = datetime.now()
            
            # Format comprehensive time info
            result = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "formatted": now.strftime("%B %d, %Y at %I:%M %p")
            }
            
            return f"Current time: {result['formatted']} ({result['day_of_week']})"
            
        except Exception as e:
            return f"Error getting time information: {str(e)}"

class FileReaderTool(BaseTool):
    """A tool for reading and summarizing text files."""
    
    name: str = "file_reader"
    description: str = "Useful for reading text files. Input should be a file path. Can read .txt, .md, .py files."
    
    def _run(self, file_path: str) -> str:
        """Execute the file reader tool."""
        try:
            # Security: only allow reading from current directory and subdirectories
            if '..' in file_path or file_path.startswith('/') or ':' in file_path:
                return "Error: For security, can only read files in current directory"
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Summarize if content is long
            if len(content) > 500:
                preview = content[:500] + "..."
                return f"File '{file_path}' content (first 500 chars):\\n{preview}\\n\\n[File has {len(content)} total characters]"
            else:
                return f"File '{file_path}' content:\\n{content}"
                
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

class SimpleToolAgent:
    """A simplified tool-enabled agent that demonstrates tool calling concepts."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the simplified tool agent."""
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
            temperature=0.1,  # Lower temperature for better tool decisions
            max_tokens=500
        )
        
        # Create tools
        self.tools = {
            "calculator": CalculatorTool(),
            "datetime": DateTimeTool(),
            "file_reader": FileReaderTool()
        }
        
        # Create prompt template for tool decision
        self.tool_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that can use tools to answer questions.

Available tools:
- calculator: For mathematical calculations (input: math expression like "2+2")
- datetime: For current date/time information (input: any time-related question)
- file_reader: For reading text files (input: file path like "README.md")

When the user asks a question:
1. Determine if you need any tools
2. If yes, respond with EXACTLY this format:
   TOOL: tool_name
   INPUT: tool_input
   
3. If no tools needed, just answer normally

Examples:
User: "What's 15 * 23?"
Response: TOOL: calculator
INPUT: 15 * 23

User: "What time is it?"  
Response: TOOL: datetime
INPUT: current time

User: "Hello how are you?"
Response: Hello! I'm doing well, thank you for asking. How can I help you today?"""),
            ("human", "{user_input}")
        ])
        
        # Create chain for tool decisions
        self.decision_chain = self.tool_prompt | self.llm
        
        print("🛠️ Simple tool agent initialized successfully!")
        print(f"📋 Available tools: {list(self.tools.keys())}")
    
    def get_response(self, user_input: str) -> str:
        """Get AI response, using tools if needed."""
        try:
            # Get LLM decision
            decision_response = self.decision_chain.invoke({"user_input": user_input})
            response_text = decision_response.content if hasattr(decision_response, 'content') else str(decision_response)
            
            # Check if LLM wants to use a tool
            if "TOOL:" in response_text:
                lines = response_text.strip().split('\n')
                tool_name = None
                tool_input = None
                
                for line in lines:
                    if line.startswith("TOOL:"):
                        tool_name = line.replace("TOOL:", "").strip()
                    elif line.startswith("INPUT:"):
                        tool_input = line.replace("INPUT:", "").strip()
                
                # Use the tool if we found both name and input
                if tool_name and tool_input and tool_name in self.tools:
                    print(f"🔧 Using {tool_name} tool with input: '{tool_input}'")
                    tool_result = self.tools[tool_name]._run(tool_input)
                    
                    # Get final response incorporating tool result
                    final_prompt = f"The user asked: '{user_input}'\nTool result: {tool_result}\nProvide a helpful response based on this information:"
                    final_response = self.llm.invoke([HumanMessage(content=final_prompt)])
                    return final_response.content if hasattr(final_response, 'content') else str(final_response)
                else:
                    return "❌ Error: Could not parse tool request properly"
            else:
                # No tool needed, return the response directly
                return response_text
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def demonstrate_tools(self):
        """Demonstrate each tool individually."""
        print("\\n🔧 Tool Demonstration")
        print("=" * 50)
        
        # Test calculator
        print("\\n🧮 Calculator Tool Demo:")
        calc_tool = CalculatorTool()
        print(f"  Input: '25 * 4 + 10'")
        print(f"  Output: {calc_tool._run('25 * 4 + 10')}")
        
        # Test datetime
        print("\\n⏰ DateTime Tool Demo:")
        dt_tool = DateTimeTool()
        print(f"  Output: {dt_tool._run('')}")
        
        # Test file reader (try to read README.md)
        print("\\n📖 File Reader Tool Demo:")
        file_tool = FileReaderTool()
        print(f"  Input: 'README.md'")
        result = file_tool._run('README.md')
        print(f"  Output: {result[:200]}...")
        
        print("\\n💡 Notice: These tools can be used individually or in combination!")
    
    def run(self):
        """Main chat loop."""
        print("\\n🤖 Simple Tool Agent - Day 7: LangChain Tools")
        print("=" * 60)
        print("This agent can intelligently decide when to use tools!")
        print("\\nTry asking:")
        print("  • 'What's 15 * 23 + 45?'")
        print("  • 'What time is it?'") 
        print("  • 'Can you read the README.md file?'")
        print("  • 'Calculate 100/7 and tell me the current date'")
        print("\\nType 'quit' to exit, '/demo' for tool demonstrations")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\\n🤖 Goodbye! Thanks for exploring LangChain tools! 🛠️")
                    break
                
                if user_input.lower() == '/demo':
                    self.demonstrate_tools()
                    continue
                
                # Skip empty input
                if not user_input:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                
                print("\\n🤖 Assistant: ", end="")
                
                # Get response using tools if needed
                response = self.get_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye! Thanks for exploring LangChain tools!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")


class ToolAgent:
    """Legacy class name for backwards compatibility."""
    def __init__(self, api_key: Optional[str] = None):
        self.agent = SimpleToolAgent(api_key)
        
    def run(self):
        self.agent.run()
        
    def get_response(self, user_input: str) -> str:
        return self.agent.get_response(user_input)
        
    def demonstrate_tools(self):
        self.agent.demonstrate_tools()

def compare_with_day4():
    """Show comparison between Day 4 (manual) and Day 7 (LangChain) tool calling."""
    print("\\n📊 Day 4 vs Day 7 Tool Calling Comparison")
    print("=" * 50)
    print("\\n**Day 4 (Manual Tool Simulation)**:")
    print("""
    # Manual decision making
    if "calculate" in user_input or any math detected:
        # Call calculator function manually
        # Parse LLM response manually
        # Handle errors manually
    
    # Complex orchestration logic
    """)
    
    print("\\n**Day 7 (LangChain Tool Calling)**:")
    print("""
    # LangChain handles everything
    tools = [CalculatorTool(), DateTimeTool(), FileReaderTool()]
    agent = create_tool_calling_agent(llm, tools, prompt)
    response = agent_executor.invoke({"input": user_input})
    """)
    
    print("\\n✅ **LangChain Tool Benefits**:")
    print("  • Automatic tool selection")
    print("  • Built-in error handling") 
    print("  • Tool result integration")
    print("  • Parallel tool calling")
    print("  • Conversation memory")
    print("  • Standardized tool interface")

def main():
    """Main function."""
    print("🚀 Starting Day 7: Tool Calling with LangChain")
    print("Learning: Agents, tools, and intelligent tool selection")
    
    # Show comparison with Day 4
    compare_with_day4()
    
    try:
        # Create and run the tool agent
        agent = ToolAgent()
        agent.run()
        
    except ImportError as e:
        print(f"\\n❌ Import Error: {e}")
        print("\\n💡 Make sure to install required packages:")
        print("   pip install langchain langchain-openai langchain-community")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()