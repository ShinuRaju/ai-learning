#!/usr/bin/env python3
"""
Day 13 - Vector Search: Question Answering with Top-K Retrieval

This script demonstrates vector search for question answering systems:
1. Building a searchable document collection
2. Converting queries and documents to embeddings
3. Performing similarity search to find relevant chunks
4. Ranking and retrieving top-k most relevant documents
5. Building a complete QA pipeline with context retrieval

Key Learning Goals:
- Vector similarity search algorithms
- Top-k retrieval strategies  
- Building searchable knowledge bases
- Query-document matching
- Foundation for RAG systems
"""

import os
import sys
import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Document:
    """Container for a document with metadata."""
    id: str
    content: str
    title: str = ""
    source: str = ""
    chunk_index: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SearchResult:
    """Container for search results."""
    document: Document
    score: float
    query: str
    rank: int
    
class VectorSearchEngine:
    """A vector search engine for document retrieval."""
    
    def __init__(self):
        """Initialize the vector search engine."""
        print("🔍 Vector Search Engine initialized!")
        print("🎯 Features: document indexing, similarity search, top-k retrieval")
        
        # OpenRouter configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Document storage
        self.documents: List[Document] = []
        self.document_embeddings: np.ndarray = None
        self.index_built = False
        
        # Sample knowledge base
        self.sample_knowledge = {
            "ai_fundamentals": [
                {
                    "title": "Machine Learning Basics",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data, learn from it, and make informed decisions or predictions."
                },
                {
                    "title": "Neural Networks",
                    "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using connectionist approaches to computation. Deep learning uses multiple layers of these networks."
                },
                {
                    "title": "Training Data", 
                    "content": "Training data is the dataset used to teach machine learning algorithms. The quality and quantity of training data directly affects model performance. Data should be representative, clean, and sufficiently large for the task."
                }
            ],
            "programming": [
                {
                    "title": "Python Fundamentals",
                    "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
                },
                {
                    "title": "Data Structures",
                    "content": "Data structures organize and store data efficiently. Common Python data structures include lists (dynamic arrays), dictionaries (hash maps), sets, and tuples. Choosing the right data structure impacts performance significantly."
                },
                {
                    "title": "Algorithms",
                    "content": "Algorithms are step-by-step procedures for solving problems. In programming, efficient algorithms reduce time and space complexity. Common algorithmic paradigms include divide-and-conquer, dynamic programming, and greedy approaches."
                }
            ],
            "databases": [
                {
                    "title": "SQL Databases",
                    "content": "SQL databases use structured query language for managing relational data. They enforce ACID properties (Atomicity, Consistency, Isolation, Durability) and use schemas to define data structure and relationships."
                },
                {
                    "title": "NoSQL Databases", 
                    "content": "NoSQL databases handle unstructured or semi-structured data. Types include document stores (MongoDB), key-value stores (Redis), column-family (Cassandra), and graph databases (Neo4j). They prioritize scalability and flexibility."
                },
                {
                    "title": "Database Indexing",
                    "content": "Database indexing improves query performance by creating shortcuts to data locations. Indexes speed up searches but require additional storage space and maintenance overhead during data modifications."
                }
            ],
            "web_development": [
                {
                    "title": "HTTP Protocol",
                    "content": "HTTP (HyperText Transfer Protocol) is the foundation of web communication. It defines how messages are formatted and transmitted between web servers and browsers. Common methods include GET, POST, PUT, DELETE."
                },
                {
                    "title": "REST APIs",
                    "content": "REST (Representational State Transfer) is an architectural style for designing web services. RESTful APIs use standard HTTP methods and status codes, are stateless, and typically exchange data in JSON format."
                },
                {
                    "title": "Frontend Frameworks",
                    "content": "Frontend frameworks like React, Vue, and Angular help build interactive user interfaces. They provide component-based architecture, state management, and virtual DOM manipulation for efficient rendering."
                }
            ]
        }
        
    def create_simple_embedding(self, text: str, model_size: int = 100) -> np.ndarray:
        """Create a simple embedding using word-based features."""
        # Preprocessing
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Feature extraction
        features = []
        
        # 1. Text statistics
        features.extend([
            len(text),
            len(words),
            len(set(words)),  # unique words
            np.mean([len(w) for w in words]) if words else 0,
            text.count('.'),
            text.count(','),
            text.count('?'),
            text.count('!')
        ])
        
        # 2. Keyword presence for different domains
        tech_keywords = ['algorithm', 'data', 'computer', 'software', 'technology', 'system', 'programming', 'code']
        ai_keywords = ['machine', 'learning', 'neural', 'intelligence', 'model', 'training', 'artificial', 'deep']
        web_keywords = ['http', 'api', 'web', 'server', 'browser', 'frontend', 'backend', 'html', 'css', 'javascript']
        db_keywords = ['database', 'sql', 'query', 'table', 'index', 'data', 'storage', 'nosql']
        
        for keyword_set in [tech_keywords, ai_keywords, web_keywords, db_keywords]:
            count = sum(1 for word in words if any(kw in word for kw in keyword_set))
            features.append(count / len(words) if words else 0)
        
        # 3. Structural features
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0
        features.extend([sentence_count, avg_sentence_length])
        
        # 4. Content type indicators
        has_definition = any(phrase in text for phrase in ['is a', 'are a', 'refers to', 'means'])
        has_example = any(phrase in text for phrase in ['example', 'such as', 'like', 'including'])
        has_process = any(phrase in text for phrase in ['step', 'process', 'method', 'procedure'])
        
        features.extend([int(has_definition), int(has_example), int(has_process)])
        
        # 5. Word frequency features (simple bag of words for common terms)
        important_words = ['learn', 'use', 'create', 'build', 'system', 'data', 'model', 'web', 'database']
        for word in important_words:
            count = sum(1 for w in words if word in w)
            features.append(count / len(words) if words else 0)
        
        # Pad or truncate to target size
        embedding = np.array(features, dtype=np.float32)
        
        if len(embedding) < model_size:
            padding = np.random.normal(0, 0.01, model_size - len(embedding))
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:model_size]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def add_document(self, content: str, title: str = "", source: str = "", 
                    chunk_index: int = 0, metadata: Dict = None) -> str:
        """Add a document to the search index."""
        doc_id = f"doc_{len(self.documents)}_{datetime.now().microsecond}"
        
        document = Document(
            id=doc_id,
            content=content,
            title=title,
            source=source,
            chunk_index=chunk_index,
            metadata=metadata or {}
        )
        
        self.documents.append(document)
        self.index_built = False  # Mark index as needing rebuild
        
        print(f"📄 Added document: {title or doc_id}")
        return doc_id
    
    def load_sample_knowledge(self):
        """Load the sample knowledge base."""
        print("📚 Loading sample knowledge base...")
        
        total_docs = 0
        for category, docs in self.sample_knowledge.items():
            for i, doc in enumerate(docs):
                self.add_document(
                    content=doc["content"],
                    title=doc["title"],
                    source=category,
                    chunk_index=i,
                    metadata={"category": category}
                )
                total_docs += 1
        
        print(f"✅ Loaded {total_docs} documents across {len(self.sample_knowledge)} categories")
    
    def build_index(self):
        """Build the search index by creating embeddings for all documents."""
        if not self.documents:
            print("❌ No documents to index")
            return
        
        print(f"🔨 Building search index for {len(self.documents)} documents...")
        
        embeddings = []
        for i, doc in enumerate(self.documents):
            # Combine title and content for embedding
            text_to_embed = f"{doc.title} {doc.content}".strip()
            embedding = self.create_simple_embedding(text_to_embed)
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.documents)} documents")
        
        self.document_embeddings = np.array(embeddings)
        self.index_built = True
        
        print(f"✅ Index built! Embedding shape: {self.document_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[SearchResult]:
        """Search for documents similar to the query."""
        if not self.index_built:
            print("🔨 Index not built, building now...")
            self.build_index()
        
        if not self.documents:
            print("❌ No documents in index")
            return []
        
        print(f"🔍 Searching for: '{query}' (top {top_k})")
        
        # Create query embedding
        query_embedding = self.create_simple_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            
            if score >= min_score:
                result = SearchResult(
                    document=self.documents[idx],
                    score=score,
                    query=query,
                    rank=rank + 1
                )
                results.append(result)
        
        print(f"✅ Found {len(results)} relevant documents")
        return results
    
    def print_search_results(self, results: List[SearchResult]):
        """Print search results in a formatted way."""
        if not results:
            print("🚫 No results found")
            return
        
        print(f"\n📊 Search Results (Found {len(results)} documents):")
        print("=" * 60)
        
        for result in results:
            doc = result.document
            print(f"\n🏆 Rank {result.rank} | Score: {result.score:.3f}")
            print(f"📝 Title: {doc.title}")
            if doc.source:
                print(f"🏷️  Category: {doc.source}")
            print(f"📄 Content: {doc.content[:200]}{'...' if len(doc.content) > 200 else ''}")
            print("-" * 60)
    
    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a question using retrieved documents."""
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        # Search for relevant documents
        results = self.search(question, top_k=top_k, min_score=0.1)
        
        if not results:
            return "❌ I couldn't find any relevant information to answer your question."
        
        # Build context from top results
        contexts = []
        for result in results:
            doc = result.document
            context = f"**{doc.title}**\n{doc.content}"
            contexts.append(context)
        
        # Simple answer generation (in a real system, you'd use an LLM here)
        answer_parts = [
            f"📚 Based on {len(results)} relevant documents:",
            "",
            "🔍 **Most relevant information:**"
        ]
        
        for i, result in enumerate(results, 1):
            doc = result.document
            answer_parts.append(f"\n{i}. **{doc.title}** (relevance: {result.score:.1%})")
            answer_parts.append(f"   {doc.content}")
        
        answer_parts.extend([
            "",
            "💡 **Key insights:**",
            "• " + results[0].document.content.split('.')[0] + ".",
            "• This information comes from the " + results[0].document.source + " domain.",
            f"• Found {len(results)} related concepts in the knowledge base."
        ])
        
        return "\n".join(answer_parts)
    
    def compare_queries(self, query1: str, query2: str):
        """Compare search results for two different queries."""
        print(f"\n🔄 Comparing Queries:")
        print(f"Query 1: '{query1}'")
        print(f"Query 2: '{query2}'")
        print("=" * 60)
        
        results1 = self.search(query1, top_k=3)
        results2 = self.search(query2, top_k=3)
        
        print(f"\n📊 Query 1 Results:")
        self.print_search_results(results1)
        
        print(f"\n📊 Query 2 Results:")
        self.print_search_results(results2)
        
        # Find common documents
        docs1_ids = {r.document.id for r in results1}
        docs2_ids = {r.document.id for r in results2}
        common_ids = docs1_ids.intersection(docs2_ids)
        
        if common_ids:
            print(f"\n🔗 Overlap: {len(common_ids)} documents found in both searches")
        else:
            print(f"\n❌ No overlap: Queries returned completely different results")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        categories = {}
        total_content_length = 0
        
        for doc in self.documents:
            cat = doc.source or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            total_content_length += len(doc.content)
        
        stats = {
            "total_documents": len(self.documents),
            "categories": categories,
            "avg_content_length": total_content_length / len(self.documents),
            "index_built": self.index_built,
            "embedding_dimension": self.document_embeddings.shape[1] if self.index_built else 0
        }
        
        return stats
    
    def export_index(self, filepath: str):
        """Export the search index to a file."""
        if not self.index_built:
            print("❌ Index not built, cannot export")
            return
        
        export_data = {
            "documents": [asdict(doc) for doc in self.documents],
            "embeddings": self.document_embeddings.tolist(),
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_documents": len(self.documents),
                "embedding_dimension": self.document_embeddings.shape[1]
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Index exported to {filepath}")
    
    def demonstrate_search_capabilities(self):
        """Demonstrate various search capabilities."""
        print("\n🎯 Search Capabilities Demonstration")
        print("=" * 50)
        
        # Load knowledge base
        self.load_sample_knowledge()
        self.build_index()
        
        # Test queries covering different topics
        test_queries = [
            "How do machine learning algorithms work?",
            "What are the best programming practices?", 
            "How to design a database?",
            "What is web development?",
            "Explain neural networks",
            "Python data structures",
            "REST API design"
        ]
        
        print("\n🧪 Testing various queries:")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            results = self.search(query, top_k=2, min_score=0.1)
            
            if results:
                for r in results:
                    print(f"   ✅ {r.document.title} ({r.score:.3f}) - {r.document.source}")
            else:
                print("   ❌ No relevant results found")
        
        # Demonstrate different search parameters
        print(f"\n🔬 Parameter Effects:")
        print("-" * 30)
        
        demo_query = "machine learning"
        
        print(f"Query: '{demo_query}'")
        print("Top-3 vs Top-5 results:")
        
        results_3 = self.search(demo_query, top_k=3)
        results_5 = self.search(demo_query, top_k=5)
        
        print(f"  Top-3: {len(results_3)} results")
        print(f"  Top-5: {len(results_5)} results")
        
        # Show score distribution
        if results_5:
            scores = [r.score for r in results_5]
            print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    def interactive_search(self):
        """Interactive search interface."""
        print("\n🔍 Interactive Vector Search")
        print("=" * 40)
        print("Commands:")
        print("  • 'search <query>' - Search documents")
        print("  • 'ask <question>' - Get detailed answer")
        print("  • 'compare <query1> | <query2>' - Compare queries")
        print("  • 'stats' - Show search statistics")
        print("  • 'load' - Load sample knowledge base")
        print("  • 'demo' - Run capability demonstration")
        print("  • 'export <file>' - Export search index")
        print("  • 'quit' - Exit")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n🔍 Search: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thanks for exploring vector search! 🚀")
                    break
                
                elif user_input.lower() == 'load':
                    self.load_sample_knowledge()
                    print("🔄 Building search index...")
                    self.build_index()
                
                elif user_input.lower() == 'demo':
                    self.demonstrate_search_capabilities()
                
                elif user_input.lower() == 'stats':
                    stats = self.get_statistics()
                    print(f"\n📊 Search Engine Statistics:")
                    for key, value in stats.items():
                        print(f"  • {key}: {value}")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        results = self.search(query, top_k=5)
                        self.print_search_results(results)
                    else:
                        print("❌ Please provide a search query")
                
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    if question:
                        answer = self.answer_question(question)
                        print(f"\n🤖 Answer:\n{answer}")
                    else:
                        print("❌ Please provide a question")
                
                elif ' | ' in user_input and user_input.startswith('compare '):
                    compare_text = user_input[8:].strip()
                    if ' | ' in compare_text:
                        query1, query2 = compare_text.split(' | ', 1)
                        self.compare_queries(query1.strip(), query2.strip())
                    else:
                        print("❌ Use format: compare <query1> | <query2>")
                
                elif user_input.startswith('export '):
                    filename = user_input[7:].strip()
                    if filename:
                        self.export_index(filename)
                    else:
                        print("❌ Please provide a filename")
                
                elif not user_input:
                    continue
                
                else:
                    print("❌ Unknown command. Use 'search <query>', 'ask <question>', 'compare <query1> | <query2>', 'stats', 'load', 'demo', 'export <file>', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy searching!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def explain_vector_search_concepts():
    """Explain key vector search concepts."""
    print("\n🎯 Vector Search Concepts")
    print("=" * 40)
    
    print("\n**What is Vector Search?**")
    print("• Finding similar documents using vector embeddings")
    print("• Documents and queries converted to numerical vectors")
    print("• Similarity calculated using mathematical distance metrics")
    print("• Top-k most similar documents returned as results")
    
    print("\n**Key Components:**")
    print("• **Document Index**: Collection of embedded documents")
    print("• **Query Processing**: Convert search query to embedding")
    print("• **Similarity Calculation**: Compare query to all documents")
    print("• **Ranking**: Sort by similarity score (cosine, euclidean, etc.)")
    print("• **Retrieval**: Return top-k most relevant documents")
    
    print("\n**Search Strategies:**")
    print("• **Exact Search**: Compare query to every document (simple, slow)")
    print("• **Approximate Search**: Use indexes for faster retrieval")
    print("• **Hybrid Search**: Combine vector + keyword search")
    print("• **Multi-Vector**: Use multiple embeddings per document")
    
    print("\n**Quality Metrics:**")
    print("• **Relevance**: How well results match query intent")
    print("• **Precision**: Percentage of relevant results returned")
    print("• **Recall**: Percentage of relevant documents found")
    print("• **Latency**: Time to return search results")
    
    print("\n**Applications:**")
    print("• **Semantic Search**: Find conceptually similar content")
    print("• **Question Answering**: Retrieve context for answers")
    print("• **Document Classification**: Group similar documents")
    print("• **Recommendation**: Suggest related items")
    print("• **RAG Systems**: Retrieve relevant knowledge for LLMs")
    
    print("\n**Production Considerations:**")
    print("• **Scalability**: Handling millions of documents")
    print("• **Index Updates**: Adding/removing documents efficiently")
    print("• **Memory Usage**: Large embeddings require significant RAM")
    print("• **Query Optimization**: Caching, pre-filtering, batching")

def main():
    """Main function."""
    print("🚀 Starting Day 13: Vector Search")
    print("Learning: Question answering with top-k document retrieval")
    
    # Explain concepts first
    explain_vector_search_concepts()
    
    try:
        # Create search engine and run
        search_engine = VectorSearchEngine()
        search_engine.interactive_search()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install numpy scikit-learn requests")
        print("\n🔄 Trying basic demo without full dependencies...")
        
        try:
            search_engine = VectorSearchEngine()
            search_engine.load_sample_knowledge()
            search_engine.build_index()
            
            # Basic test
            results = search_engine.search("machine learning", top_k=3)
            search_engine.print_search_results(results)
            
        except Exception as e2:
            print(f"❌ Basic demo also failed: {e2}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()