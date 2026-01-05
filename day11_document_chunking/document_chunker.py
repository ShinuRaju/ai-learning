#!/usr/bin/env python3
"""
Day 11 - Document Chunking: Load and Chunk Documents

This script demonstrates document chunking strategies essential for RAG systems:
1. Loading documents from various sources
2. Different chunking strategies and their trade-offs
3. Chunk overlap and size optimization
4. Content-aware chunking for different document types
5. Preparing documents for embedding and retrieval

Key Learning Goals:
- Understanding chunking strategies for RAG
- Document loading and preprocessing
- Chunk size vs context trade-offs
- Content-aware splitting techniques
- Preparing for embeddings and vector storage
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import re
import json

# LangChain imports for document processing
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document

class DocumentChunker:
    """A comprehensive document chunking system with multiple strategies."""
    
    def __init__(self):
        """Initialize the document chunker with various splitters."""
        print("📄 Document Chunker initialized!")
        print("🔧 Available chunking strategies: recursive, character, token, markdown, code")
        
        # Initialize different splitters
        self.splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            "character": CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n\n"
            ),
            "token": TokenTextSplitter(
                chunk_size=250,  # Tokens, not characters
                chunk_overlap=50
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "code": PythonCodeTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
        }
    
    def load_document_from_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """Create a document from text string."""
        return Document(
            page_content=text,
            metadata=metadata or {"source": "manual_input"}
        )
    
    def load_document_from_file(self, file_path: str) -> Optional[Document]:
        """Load a document from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return Document(
                    page_content=content,
                    metadata={"source": file_path, "type": "file"}
                )
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return None
    
    def chunk_document(self, document: Document, strategy: str = "recursive") -> List[Document]:
        """Chunk a document using the specified strategy."""
        if strategy not in self.splitters:
            print(f"❌ Unknown strategy: {strategy}. Using 'recursive' instead.")
            strategy = "recursive"
        
        print(f"🔄 Chunking with {strategy} strategy...")
        
        try:
            splitter = self.splitters[strategy]
            chunks = splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_strategy": strategy,
                    "chunk_size": len(chunk.page_content),
                    "total_chunks": len(chunks)
                })
            
            print(f"✅ Created {len(chunks)} chunks using {strategy} strategy")
            return chunks
            
        except Exception as e:
            print(f"❌ Error chunking document: {e}")
            return []
    
    def analyze_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze chunk statistics and quality."""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        analysis = {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "strategy_used": chunks[0].metadata.get("chunk_strategy", "unknown"),
            "size_distribution": {
                "small (< 500)": len([s for s in chunk_sizes if s < 500]),
                "medium (500-1000)": len([s for s in chunk_sizes if 500 <= s <= 1000]),
                "large (> 1000)": len([s for s in chunk_sizes if s > 1000])
            }
        }
        
        return analysis
    
    def compare_strategies(self, document: Document, strategies: List[str] = None) -> Dict[str, Any]:
        """Compare different chunking strategies on the same document."""
        if strategies is None:
            strategies = ["recursive", "character", "token"]
        
        print(f"📊 Comparing chunking strategies: {', '.join(strategies)}")
        
        comparison = {}
        
        for strategy in strategies:
            if strategy in self.splitters:
                chunks = self.chunk_document(document, strategy)
                analysis = self.analyze_chunks(chunks)
                comparison[strategy] = analysis
        
        return comparison
    
    def demonstrate_overlap_effect(self, document: Document) -> None:
        """Demonstrate the effect of different overlap sizes."""
        print("\n🔍 Demonstrating Chunk Overlap Effects")
        print("=" * 50)
        
        overlap_sizes = [0, 100, 200, 400]
        
        for overlap in overlap_sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = splitter.split_documents([document])
            analysis = self.analyze_chunks(chunks)
            
            print(f"\nOverlap: {overlap} characters")
            print(f"  Chunks created: {analysis['total_chunks']}")
            print(f"  Average size: {analysis['average_chunk_size']:.0f} chars")
            
            if len(chunks) >= 2:
                # Show actual overlap between first two chunks
                chunk1 = chunks[0].page_content
                chunk2 = chunks[1].page_content
                
                # Simple overlap detection
                overlap_found = 0
                for i in range(min(len(chunk1), overlap + 100)):
                    suffix = chunk1[-(i+1):]
                    if chunk2.startswith(suffix):
                        overlap_found = len(suffix)
                        break
                
                print(f"  Actual overlap: ~{overlap_found} chars")
    
    def create_sample_documents(self) -> Dict[str, Document]:
        """Create sample documents for demonstration."""
        samples = {}
        
        # Sample 1: Technical article
        technical_text = """
# Introduction to Artificial Intelligence

Artificial Intelligence (AI) is a rapidly evolving field that encompasses machine learning, natural language processing, computer vision, and robotics. The goal of AI is to create systems that can perform tasks that typically require human intelligence.

## Machine Learning Fundamentals

Machine learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. There are three main types of machine learning:

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common algorithms include:
- Linear regression for predicting continuous values
- Logistic regression for classification tasks
- Decision trees for both regression and classification
- Neural networks for complex pattern recognition

### Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples. Key techniques include:
- Clustering algorithms like K-means and hierarchical clustering
- Dimensionality reduction techniques like PCA and t-SNE
- Anomaly detection for identifying outliers

### Reinforcement Learning
Reinforcement learning learns optimal actions through trial and error interactions with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time.

## Natural Language Processing

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Modern NLP applications include:

- Language translation systems
- Chatbots and virtual assistants
- Sentiment analysis for social media monitoring
- Text summarization for document processing
- Named entity recognition for information extraction

## Applications and Future Directions

AI technology is being applied across various industries:

Healthcare: AI assists in medical diagnosis, drug discovery, and personalized treatment plans. Medical imaging analysis using deep learning has shown remarkable accuracy in detecting diseases like cancer.

Finance: Algorithmic trading, fraud detection, and credit scoring rely heavily on AI techniques. Machine learning models analyze market patterns and transaction data to make informed decisions.

Transportation: Autonomous vehicles use computer vision, sensor fusion, and deep learning to navigate safely. AI optimizes route planning and traffic management systems.

The future of AI holds promise for even more sophisticated applications, including artificial general intelligence (AGI) that could match or exceed human cognitive abilities across all domains.
"""
        
        samples["technical"] = self.load_document_from_text(
            technical_text,
            {"title": "Introduction to AI", "type": "technical_article", "domain": "artificial_intelligence"}
        )
        
        # Sample 2: Financial data
        financial_text = """
Quarterly Financial Report - Q3 2024

Executive Summary:
Our company achieved strong financial performance in Q3 2024, with revenue growth of 15% year-over-year and improved profitability across all business segments.

Revenue Analysis:
Total revenue for Q3 2024 reached $125.6 million, compared to $109.2 million in Q3 2023. This 15% increase was driven by:
- Product sales: $89.4 million (up 12% YoY)
- Service revenue: $36.2 million (up 22% YoY)

The service revenue growth reflects our strategic focus on recurring revenue streams and customer success initiatives.

Expense Breakdown:
Operating expenses totaled $98.7 million, representing a 9% increase from the previous year:
- Personnel costs: $54.3 million (55% of total expenses)
- Technology investments: $18.9 million (19% of total expenses)
- Marketing and sales: $15.2 million (15% of total expenses)
- General and administrative: $10.3 million (11% of total expenses)

Profitability Metrics:
Gross profit margin improved to 68%, up from 65% in Q3 2023. This improvement reflects operational efficiencies and favorable product mix changes.

EBITDA reached $26.9 million, representing a 21% margin, compared to 19% in the prior year quarter.

Net income was $18.4 million, or $0.52 per share, compared to $14.1 million, or $0.41 per share, in Q3 2023.

Cash Flow and Balance Sheet:
Cash flow from operations was $22.1 million, providing strong liquidity for growth investments.
Cash and cash equivalents totaled $47.8 million at quarter end.
Total debt decreased to $23.4 million, improving our debt-to-equity ratio to 0.3.

Market Outlook:
We remain optimistic about Q4 2024 and expect continued revenue growth driven by new product launches and expansion into international markets.
"""
        
        samples["financial"] = self.load_document_from_text(
            financial_text,
            {"title": "Q3 2024 Financial Report", "type": "financial_report", "quarter": "Q3_2024"}
        )
        
        # Sample 3: Code documentation
        code_text = """
# Document Processing System

## Overview
This module provides utilities for processing and analyzing documents in various formats.

```python
class DocumentProcessor:
    '''Main class for document processing operations.'''
    
    def __init__(self, config_path: str = None):
        '''Initialize the document processor.
        
        Args:
            config_path: Path to configuration file
        '''
        self.config = self._load_config(config_path)
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']
    
    def process_document(self, file_path: str, output_format: str = 'json'):
        '''Process a document and extract structured information.
        
        Args:
            file_path: Path to the input document
            output_format: Desired output format ('json', 'xml', 'csv')
            
        Returns:
            Processed document data in specified format
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            UnsupportedFormatError: If file format is not supported
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.supported_formats:
            raise UnsupportedFormatError(f"Format {file_extension} not supported")
        
        # Load document content
        content = self._load_content(file_path)
        
        # Extract metadata
        metadata = self._extract_metadata(content)
        
        # Process content
        processed_data = self._process_content(content, metadata)
        
        # Format output
        return self._format_output(processed_data, output_format)
    
    def _load_content(self, file_path: str) -> str:
        '''Load document content based on file type.'''
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            return self._load_text_file(file_path)
        elif file_extension == '.pdf':
            return self._load_pdf_file(file_path)
        elif file_extension == '.docx':
            return self._load_docx_file(file_path)
        elif file_extension == '.md':
            return self._load_markdown_file(file_path)
    
    def _extract_metadata(self, content: str) -> dict:
        '''Extract metadata from document content.'''
        metadata = {
            'word_count': len(content.split()),
            'char_count': len(content),
            'line_count': content.count('\\n') + 1,
            'paragraph_count': len([p for p in content.split('\\n\\n') if p.strip()])
        }
        return metadata

# Usage Example
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    try:
        result = processor.process_document('sample.pdf', 'json')
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing document: {e}")
```

## Configuration

The system uses a YAML configuration file for customization:

```yaml
# config.yaml
processing:
  max_file_size: 50MB
  timeout_seconds: 300
  
output:
  include_metadata: true
  preserve_formatting: false
  
formats:
  json:
    indent: 2
    sort_keys: true
  xml:
    encoding: utf-8
    pretty_print: true
```

## Error Handling

The system implements comprehensive error handling:

1. **File Access Errors**: Handles missing files, permission issues
2. **Format Errors**: Validates file formats before processing  
3. **Memory Errors**: Manages large file processing
4. **Timeout Errors**: Prevents indefinite processing
"""
        
        samples["code"] = self.load_document_from_text(
            code_text,
            {"title": "Document Processing System", "type": "code_documentation", "language": "python"}
        )
        
        return samples
    
    def run_demonstrations(self):
        """Run comprehensive chunking demonstrations."""
        print("\n📄 Document Chunking Demonstrations - Day 11")
        print("=" * 60)
        
        # Create sample documents
        print("📋 Creating sample documents...")
        samples = self.create_sample_documents()
        
        # Demo 1: Basic chunking comparison
        print("\n🔍 Demo 1: Chunking Strategy Comparison")
        print("-" * 40)
        
        tech_doc = samples["technical"]
        comparison = self.compare_strategies(tech_doc, ["recursive", "character", "token"])
        
        for strategy, analysis in comparison.items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Chunks: {analysis['total_chunks']}")
            print(f"  Avg size: {analysis['average_chunk_size']:.0f} chars")
            print(f"  Size range: {analysis['min_chunk_size']}-{analysis['max_chunk_size']}")
        
        # Demo 2: Content-aware chunking
        print("\n\n🎯 Demo 2: Content-Aware Chunking")
        print("-" * 40)
        
        # Demonstrate different splitters for different content types
        content_types = [
            ("Technical Article", samples["technical"], "recursive"),
            ("Financial Report", samples["financial"], "character"),
            ("Code Documentation", samples["code"], "code")
        ]
        
        for name, doc, strategy in content_types:
            print(f"\n{name} ({strategy}):")
            chunks = self.chunk_document(doc, strategy)
            analysis = self.analyze_chunks(chunks)
            
            print(f"  Chunks created: {analysis['total_chunks']}")
            print(f"  Average size: {analysis['average_chunk_size']:.0f} chars")
            print(f"  Distribution: {analysis['size_distribution']}")
        
        # Demo 3: Overlap effect
        self.demonstrate_overlap_effect(samples["technical"])
        
        # Demo 4: Chunk content preview
        print("\n\n📖 Demo 4: Chunk Content Preview")
        print("-" * 40)
        
        chunks = self.chunk_document(samples["financial"], "recursive")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Size: {len(chunk.page_content)} chars")
            print(f"  Preview: {chunk.page_content[:200]}...")
            print(f"  Metadata: {chunk.metadata}")
    
    def interactive_chunking(self):
        """Interactive chunking interface."""
        print("\n🎮 Interactive Document Chunking")
        print("=" * 50)
        print("Commands:")
        print("  • 'sample <type>' - Load sample document (technical/financial/code)")
        print("  • 'text <your_text>' - Process custom text")
        print("  • 'chunk <strategy>' - Chunk current document")
        print("  • 'analyze' - Analyze current chunks")
        print("  • 'compare' - Compare all strategies")
        print("  • 'demo' - Run full demonstrations")
        print("  • 'quit' - Exit")
        print("-" * 50)
        
        current_document = None
        current_chunks = []
        
        while True:
            try:
                user_input = input("\n📄 Chunker: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n📄 Thanks for exploring document chunking! 🚀")
                    break
                
                elif user_input.lower() == 'demo':
                    self.run_demonstrations()
                
                elif user_input.startswith('sample '):
                    doc_type = user_input[7:].strip().lower()
                    samples = self.create_sample_documents()
                    
                    if doc_type in samples:
                        current_document = samples[doc_type]
                        print(f"✅ Loaded {doc_type} sample document")
                        print(f"   Length: {len(current_document.page_content)} chars")
                        print(f"   Title: {current_document.metadata.get('title', 'Unknown')}")
                    else:
                        print(f"❌ Unknown sample type. Available: {list(samples.keys())}")
                
                elif user_input.startswith('text '):
                    text = user_input[5:].strip()
                    if text:
                        current_document = self.load_document_from_text(text, {"source": "user_input"})
                        print(f"✅ Loaded custom text ({len(text)} chars)")
                    else:
                        print("❌ Please provide text after 'text' command")
                
                elif user_input.startswith('chunk '):
                    if not current_document:
                        print("❌ No document loaded. Use 'sample' or 'text' first.")
                        continue
                    
                    strategy = user_input[6:].strip()
                    current_chunks = self.chunk_document(current_document, strategy)
                    print(f"✅ Document chunked into {len(current_chunks)} pieces")
                
                elif user_input.lower() == 'analyze':
                    if not current_chunks:
                        print("❌ No chunks to analyze. Use 'chunk' first.")
                        continue
                    
                    analysis = self.analyze_chunks(current_chunks)
                    print("\n📊 Chunk Analysis:")
                    for key, value in analysis.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"  {key}: {value}")
                
                elif user_input.lower() == 'compare':
                    if not current_document:
                        print("❌ No document loaded. Use 'sample' or 'text' first.")
                        continue
                    
                    comparison = self.compare_strategies(current_document)
                    print("\n📊 Strategy Comparison:")
                    for strategy, analysis in comparison.items():
                        print(f"\n{strategy.upper()}:")
                        print(f"  Chunks: {analysis['total_chunks']}")
                        print(f"  Avg size: {analysis['average_chunk_size']:.0f} chars")
                        print(f"  Range: {analysis['min_chunk_size']}-{analysis['max_chunk_size']}")
                
                elif not user_input:
                    continue
                
                else:
                    print("❌ Unknown command. Type 'demo', 'sample <type>', 'text <content>', 'chunk <strategy>', 'analyze', 'compare', or 'quit'")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy chunking!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def explain_chunking_concepts():
    """Explain key chunking concepts and trade-offs."""
    print("\n🧠 Document Chunking Concepts")
    print("=" * 40)
    
    print("\n**Why Chunking Matters:**")
    print("• LLM context windows have limits (4K-128K tokens)")
    print("• Large documents exceed these limits")
    print("• Retrieval works better with focused chunks")
    print("• Embedding quality improves with optimal chunk size")
    
    print("\n**Key Trade-offs:**")
    print("• **Chunk Size**: Large chunks = more context, but less precise retrieval")
    print("• **Overlap**: More overlap = better context preservation, but more storage")
    print("• **Strategy**: Content-aware splitting vs. simple character limits")
    
    print("\n**Optimal Chunk Sizes:**")
    print("• **Embedding models**: 200-500 tokens")
    print("• **Retrieval systems**: 500-1000 tokens") 
    print("• **Question answering**: 1000-2000 tokens")
    
    print("\n**Chunking Strategies:**")
    print("• **Recursive**: Best general-purpose splitter")
    print("• **Character**: Simple, predictable splits")
    print("• **Token**: Precise for LLM limits")
    print("• **Content-aware**: Respects document structure")

def main():
    """Main function."""
    print("🚀 Starting Day 11: Document Chunking")
    print("Learning: Chunking strategies for RAG systems")
    
    # Explain concepts
    explain_chunking_concepts()
    
    try:
        # Create chunker and run
        chunker = DocumentChunker()
        chunker.interactive_chunking()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Make sure to install required packages:")
        print("   pip install langchain tiktoken")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()