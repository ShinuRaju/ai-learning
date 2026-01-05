# Day 2 - Prompt Control: Finance Advisor

## 🎯 Project Overview

Build a role-based AI assistant that acts as a professional finance advisor using system prompts to control behavior and personality.

## 🧠 Learning Objectives

- **System vs User Prompts**: Understand how system prompts shape AI behavior
- **Prompt Discipline**: Learn to craft effective, specific prompts
- **Role-based AI**: Make AI assume specific personas and expertise

## 🔧 What You'll Build

A CLI finance advisor that:

- Acts as a professional financial consultant
- Maintains character consistency
- Provides structured, helpful advice
- Asks clarifying questions
- Remembers conversation context

## 🚀 How to Run

```bash
cd day02_prompt_control
python finance_advisor.py
```

## 💡 Key Concepts

### System Prompt

The system prompt defines the AI's role, personality, and behavior guidelines:

```
You are a professional finance advisor with 15+ years of experience...
```

### Conversation Flow

1. **System message** - Sets the role and rules
2. **User message** - The human's question/input
3. **Assistant message** - AI's response following system guidelines

### Prompt Discipline

- Be specific about the role
- Set clear guidelines and boundaries
- Define output format expectations
- Include behavioral constraints

## 🔍 Try These Examples

### Basic Questions

- "How much should I save each month?"
- "What's the difference between 401k and IRA?"
- "I have $5000 to invest, what should I do?"

### Complex Scenarios

- "I'm 25, make $60k, have $10k debt. Help me plan."
- "Should I buy a house or keep renting?"
- "How do I start investing with only $100?"

## 🎓 What You're Learning

### 1. System Prompts Control Behavior

Notice how the AI:

- Stays in character as a finance advisor
- Asks clarifying questions
- Focuses on practical advice
- Avoids risky recommendations

### 2. Prompt Engineering Principles

- **Role definition**: "You are a professional finance advisor"
- **Guidelines**: Specific do's and don'ts
- **Output format**: How responses should be structured
- **Boundaries**: What the AI should/shouldn't do

### 3. Conversation Context

The system maintains context by:

- Storing conversation history
- Including it in each API call
- Limiting history to prevent token overflow

## 🚀 Next Steps

Tomorrow (Day 3), you'll learn about **structured output** - making the AI return data in specific JSON formats for programmatic use.

## 💭 Reflection Questions

- How does the system prompt change the AI's responses?
- What happens if you modify the advisor's "experience level"?
- How could you adapt this pattern for other roles (teacher, coach, etc.)?
