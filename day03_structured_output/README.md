# Day 3 - Structured Output: Expense Categorizer

## 🎯 Learning Goals

- **Structured output from LLMs**: Getting consistent JSON responses
- **JSON validation**: Parsing and validating LLM outputs
- **Prompt discipline**: Crafting prompts for reliable formatting

## 🔍 Project Overview

Build an expense categorizer that takes natural language expense descriptions and returns structured JSON with:

- `category`: One of 12 predefined expense categories
- `confidence`: Float score (0.0-1.0) indicating classification certainty
- `reasoning`: Brief explanation of the categorization decision

## 📋 Features

### Core Functionality

- 🏷️ **12 Expense Categories**: food_dining, transportation, entertainment, utilities, healthcare, shopping_retail, groceries, rent_housing, travel, education, insurance, other
- 📊 **Confidence Scoring**: Visual confidence bars and numeric scores
- 🧠 **Reasoning**: LLM explains its categorization logic
- ✅ **JSON Validation**: Strict parsing and validation of LLM responses

### Example Input/Output

```
Input: "Starbucks coffee and pastry $8.75"

Output:
{
  "category": "food_dining",
  "confidence": 0.9,
  "reasoning": "Coffee shop purchase with food items"
}
```

## 🚀 How to Run

1. **Set up your environment:**

   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   # On Windows: set OPENROUTER_API_KEY=your-api-key-here
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the categorizer:**

   ```bash
   python expense_categorizer.py
   ```

4. **Try some examples:**
   - "Netflix monthly subscription $15.99"
   - "Uber ride to downtown $23.40"
   - "Grocery shopping at Target $89.32"
   - "Electric bill payment $145.67"

## 🔧 Key Technical Concepts

### 1. Structured Prompt Engineering

```python
def create_categorization_prompt(self, expense_description: str) -> str:
    return f"""You are an expense categorization assistant.

    You MUST respond with valid JSON in exactly this format:
    {{
        "category": "category_name",
        "confidence": confidence_score,
        "reasoning": "brief explanation"
    }}
    """
```

### 2. JSON Validation Pipeline

- Parse JSON response from LLM
- Validate required keys exist
- Check category is in allowed list
- Verify confidence score is in valid range (0.0-1.0)
- Handle parsing errors gracefully

### 3. Error Handling

- Malformed JSON responses → fallback to "other" category
- Invalid categories → validation error and fallback
- API errors → graceful degradation with error messaging

## 📚 What You'll Learn

### Day 3 Focus Areas:

1. **Structured Output Patterns**: How to get consistent, parseable responses from LLMs
2. **JSON Schema Thinking**: Designing data structures for AI responses
3. **Validation Mindset**: Always validate and handle malformed AI outputs
4. **Prompt Precision**: Crafting prompts that minimize ambiguity
5. **Error Recovery**: Building robust systems that handle AI unpredictability

### Key Insights:

- **Temperature matters**: Low temperature (0.1) for consistent structured outputs
- **Examples work**: Providing JSON examples in prompts improves compliance
- **Validation is critical**: Never trust LLM outputs without validation
- **Fallbacks save you**: Always have a default response for failures

## 🎓 Extensions to Try

1. **Batch Processing**: Categorize multiple expenses from a CSV file
2. **Custom Categories**: Allow users to define their own expense categories
3. **Learning Mode**: Store results and improve categorization over time
4. **Confidence Tuning**: Experiment with different confidence scoring methods
5. **Multi-model Comparison**: Compare GPT-3.5 vs GPT-4 accuracy

## 🔗 Next Steps

Day 4 will build on this structured output foundation to introduce **Tool Simulation** - teaching the LLM to decide when to use external tools like calculators.

The JSON validation skills from today are essential for:

- Tool calling (validating tool parameters)
- Agent responses (structured decision making)
- RAG systems (structured retrieval metadata)
- MCP protocols (standardized data exchange)

---

_Building consistent, structured AI outputs is the foundation of reliable AI systems! 🏗️_
