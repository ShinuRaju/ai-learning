#!/usr/bin/env python3
"""
Day 12 - Embedding Vectors: Create and Explore Vector Embeddings

This script demonstrates embedding vectors and their applications in AI systems:
1. Creating embeddings from text using different models
2. Understanding vector similarity and distance metrics
3. Semantic search and text comparison
4. Visualizing embeddings in 2D space
5. Practical applications for RAG systems

Key Learning Goals:
- Understanding what embeddings represent
- Different embedding models and their characteristics  
- Vector similarity calculations
- Preparing text for semantic search
- Foundation for vector databases
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import json
import requests
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EmbeddingResult:
    """Container for embedding results with metadata."""
    text: str
    vector: np.ndarray
    model: str
    timestamp: str
    similarity_scores: Dict[str, float] = None

class EmbeddingExplorer:
    """A comprehensive embedding exploration system."""
    
    def __init__(self):
        """Initialize the embedding explorer."""
        print("🧠 Embedding Explorer initialized!")
        print("🔧 Available features: text embeddings, similarity search, visualization")
        
        # OpenRouter API configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            print("⚠️  OPENROUTER_API_KEY not found. Using local embeddings only.")
        
        # Sample texts for demonstration
        self.sample_texts = {
            "technology": [
                "Machine learning algorithms can learn patterns from data",
                "Neural networks are inspired by biological brain structures", 
                "Deep learning uses multiple layers of artificial neurons",
                "Computer vision helps machines understand images",
                "Natural language processing enables AI to understand text"
            ],
            "cooking": [
                "Chopping vegetables is essential for good cooking",
                "Seasoning food properly enhances flavor profiles",
                "Fresh ingredients make the biggest difference in taste",
                "Proper cooking temperature prevents food safety issues",
                "Knife skills are fundamental for efficient food preparation"
            ],
            "travel": [
                "Exploring new cultures broadens your perspective on life",
                "Mountain hiking offers breathtaking views and exercise",
                "Beach vacations provide relaxation and vitamin D",
                "City tours reveal historical architecture and local food",
                "Adventure travel builds confidence and unforgettable memories"
            ],
            "finance": [
                "Diversified investment portfolios reduce financial risk",
                "Compound interest grows wealth exponentially over time",
                "Emergency funds provide security during unexpected events",
                "Budget planning helps achieve long-term financial goals",
                "Market volatility requires patience and strategic thinking"
            ]
        }
        
        # Storage for embeddings
        self.embeddings_cache = {}
        
    def create_simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple word-based embedding for demonstration."""
        # Simple word frequency + position based embedding
        words = text.lower().split()
        
        # Create basic features
        features = []
        
        # 1. Text length features
        features.extend([
            len(text),                    # Character count
            len(words),                   # Word count
            np.mean([len(w) for w in words]) if words else 0,  # Avg word length
            text.count('.'),              # Sentence count
            text.count(',')               # Comma count
        ])
        
        # 2. Keyword presence (simple semantic features)
        tech_words = ['machine', 'learning', 'neural', 'algorithm', 'data', 'ai', 'computer', 'technology']
        cooking_words = ['cooking', 'food', 'recipe', 'kitchen', 'flavor', 'ingredient', 'cook', 'taste']
        travel_words = ['travel', 'trip', 'vacation', 'city', 'mountain', 'beach', 'adventure', 'culture']
        finance_words = ['money', 'investment', 'financial', 'budget', 'market', 'wealth', 'portfolio', 'economy']
        
        for category_words in [tech_words, cooking_words, travel_words, finance_words]:
            count = sum(1 for word in words if any(kw in word for kw in category_words))
            features.append(count / len(words) if words else 0)
        
        # 3. Part-of-speech approximation (simple heuristics)
        common_verbs = ['is', 'are', 'can', 'help', 'make', 'provide', 'build', 'grow', 'learn']
        common_adjectives = ['good', 'new', 'best', 'important', 'effective', 'fresh', 'proper', 'financial']
        
        verb_count = sum(1 for word in words if word in common_verbs)
        adj_count = sum(1 for word in words if word in common_adjectives)
        
        features.extend([
            verb_count / len(words) if words else 0,
            adj_count / len(words) if words else 0
        ])
        
        # 4. Positional features
        if words:
            first_word_len = len(words[0])
            last_word_len = len(words[-1])
        else:
            first_word_len = last_word_len = 0
            
        features.extend([first_word_len, last_word_len])
        
        # Normalize and pad to standard dimension
        embedding = np.array(features, dtype=np.float32)
        
        # Pad to 50 dimensions
        target_dim = 50
        if len(embedding) < target_dim:
            padding = np.zeros(target_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:target_dim]
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def create_openrouter_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create embedding using OpenRouter API (placeholder - not all models support embeddings)."""
        # Note: This is a placeholder as OpenRouter primarily focuses on chat models
        # In practice, you'd use dedicated embedding APIs like OpenAI's text-embedding-ada-002
        print("📝 Note: Using local embedding model (OpenRouter doesn't provide embedding endpoints)")
        return self.create_simple_embedding(text)
    
    def embed_text(self, text: str, model: str = "local") -> EmbeddingResult:
        """Create an embedding for the given text."""
        print(f"🔄 Creating embedding for: '{text[:50]}...'")
        
        if model == "local":
            vector = self.create_simple_embedding(text)
        else:
            # In a real implementation, you'd call different embedding APIs
            vector = self.create_simple_embedding(text)
        
        result = EmbeddingResult(
            text=text,
            vector=vector,
            model=model,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"✅ Created {len(vector)}-dimensional embedding")
        return result
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                           metric: str = "cosine") -> float:
        """Calculate similarity between two embeddings."""
        if metric == "cosine":
            # Cosine similarity: measures angle between vectors
            return cosine_similarity([embedding1], [embedding2])[0][0]
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1 / (1 + distance)  # Convert distance to similarity
        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_most_similar(self, query_text: str, candidates: List[str], 
                         top_k: int = 3, model: str = "local") -> List[Tuple[str, float]]:
        """Find most similar texts to the query."""
        print(f"🔍 Finding {top_k} most similar texts to: '{query_text}'")
        
        # Create query embedding
        query_embedding = self.embed_text(query_text, model)
        
        # Calculate similarities
        similarities = []
        for candidate in candidates:
            candidate_embedding = self.embed_text(candidate, model)
            similarity = self.calculate_similarity(
                query_embedding.vector, 
                candidate_embedding.vector
            )
            similarities.append((candidate, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Found {len(similarities)} similarities")
        return similarities[:top_k]
    
    def embed_categories(self, categories: Dict[str, List[str]]) -> Dict[str, List[EmbeddingResult]]:
        """Create embeddings for all texts in categories."""
        print(f"🔄 Creating embeddings for {len(categories)} categories")
        
        embedded_categories = {}
        total_texts = sum(len(texts) for texts in categories.values())
        processed = 0
        
        for category, texts in categories.items():
            embedded_texts = []
            for text in texts:
                embedding = self.embed_text(text, "local")
                embedded_texts.append(embedding)
                processed += 1
                print(f"  Progress: {processed}/{total_texts} texts processed")
            
            embedded_categories[category] = embedded_texts
        
        print(f"✅ Created embeddings for {processed} texts")
        return embedded_categories
    
    def create_similarity_matrix(self, embeddings: List[EmbeddingResult]) -> np.ndarray:
        """Create a similarity matrix for visualization."""
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.calculate_similarity(
                        embeddings[i].vector,
                        embeddings[j].vector
                    )
                    similarity_matrix[i][j] = sim
        
        return similarity_matrix
    
    def visualize_embeddings_2d(self, embedded_categories: Dict[str, List[EmbeddingResult]]):
        """Visualize embeddings in 2D using PCA."""
        print("📊 Creating 2D visualization of embeddings...")
        
        # Prepare data for PCA
        all_vectors = []
        labels = []
        colors = []
        color_map = {'technology': 'blue', 'cooking': 'red', 'travel': 'green', 'finance': 'orange'}
        
        for category, embeddings in embedded_categories.items():
            for embedding in embeddings:
                all_vectors.append(embedding.vector)
                labels.append(f"{category[:4]}_{len(labels)}")
                colors.append(color_map.get(category, 'gray'))
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(all_vectors)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot points by category
        for category, color in color_map.items():
            cat_indices = [i for i, label in enumerate(labels) if label.startswith(category[:4])]
            if cat_indices:
                cat_vectors = vectors_2d[cat_indices]
                plt.scatter(cat_vectors[:, 0], cat_vectors[:, 1], 
                           c=color, label=category.title(), alpha=0.7, s=100)
        
        plt.title("Embedding Visualization (2D PCA)", fontsize=16, fontweight='bold')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text labels for some points
        for i, (x, y) in enumerate(vectors_2d):
            if i % 3 == 0:  # Show every 3rd label to avoid clutter
                plt.annotate(labels[i], (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        print("✅ 2D visualization created")
    
    def visualize_similarity_heatmap(self, embedded_categories: Dict[str, List[EmbeddingResult]]):
        """Create a heatmap of similarities between categories."""
        print("🔥 Creating similarity heatmap...")
        
        # Calculate average similarities between categories
        category_names = list(embedded_categories.keys())
        n_categories = len(category_names)
        similarity_matrix = np.zeros((n_categories, n_categories))
        
        for i, cat1 in enumerate(category_names):
            for j, cat2 in enumerate(category_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate average similarity between categories
                    similarities = []
                    for emb1 in embedded_categories[cat1]:
                        for emb2 in embedded_categories[cat2]:
                            sim = self.calculate_similarity(emb1.vector, emb2.vector)
                            similarities.append(sim)
                    similarity_matrix[i][j] = np.mean(similarities)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, 
                    xticklabels=[cat.title() for cat in category_names],
                    yticklabels=[cat.title() for cat in category_names],
                    annot=True, fmt='.3f', cmap='RdYlBu_r',
                    center=0.5, vmin=0, vmax=1)
        
        plt.title("Category Similarity Heatmap", fontsize=16, fontweight='bold')
        plt.xlabel("Categories", fontsize=12)
        plt.ylabel("Categories", fontsize=12)
        plt.tight_layout()
        plt.show()
        print("✅ Similarity heatmap created")
    
    def demonstrate_semantic_search(self):
        """Demonstrate semantic search capabilities."""
        print("\n🔍 Semantic Search Demonstration")
        print("=" * 50)
        
        # Create a combined dataset
        all_texts = []
        text_categories = {}
        
        for category, texts in self.sample_texts.items():
            all_texts.extend(texts)
            for text in texts:
                text_categories[text] = category
        
        # Test queries
        test_queries = [
            "How do computers learn from information?",
            "What makes food taste better?", 
            "Planning a vacation in the mountains",
            "Building wealth through investments"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: '{query}'")
            print("-" * 30)
            
            # Find similar texts
            results = self.find_most_similar(query, all_texts, top_k=3)
            
            for rank, (text, similarity) in enumerate(results, 1):
                category = text_categories[text]
                print(f"  {rank}. [{category.upper()}] {text}")
                print(f"     Similarity: {similarity:.3f}")
    
    def run_comprehensive_demo(self):
        """Run a comprehensive embedding demonstration."""
        print("\n🧠 Comprehensive Embedding Demonstration")
        print("=" * 60)
        
        # 1. Create embeddings for all categories
        print("\n📊 Step 1: Creating Embeddings")
        print("-" * 30)
        embedded_categories = self.embed_categories(self.sample_texts)
        
        # 2. Analyze embedding properties
        print("\n🔬 Step 2: Analyzing Embedding Properties")
        print("-" * 30)
        
        for category, embeddings in embedded_categories.items():
            vectors = [emb.vector for emb in embeddings]
            avg_norm = np.mean([np.linalg.norm(v) for v in vectors])
            std_norm = np.std([np.linalg.norm(v) for v in vectors])
            
            print(f"{category.title()}:")
            print(f"  Embeddings: {len(embeddings)}")
            print(f"  Avg norm: {avg_norm:.3f} ± {std_norm:.3f}")
            print(f"  Dimensions: {len(vectors[0]) if vectors else 0}")
        
        # 3. Semantic search demo
        self.demonstrate_semantic_search()
        
        # 4. Visualizations
        print("\n📈 Step 3: Creating Visualizations")
        print("-" * 30)
        
        try:
            self.visualize_embeddings_2d(embedded_categories)
            self.visualize_similarity_heatmap(embedded_categories)
        except Exception as e:
            print(f"⚠️  Visualization skipped: {e}")
            print("💡 Install matplotlib and seaborn for visualizations")
        
        print("\n✅ Comprehensive demonstration completed!")
    
    def interactive_embedding_explorer(self):
        """Interactive embedding exploration interface."""
        print("\n🎮 Interactive Embedding Explorer")
        print("=" * 50)
        print("Commands:")
        print("  • 'embed <text>' - Create embedding for custom text")
        print("  • 'search <query>' - Find similar texts")
        print("  • 'compare <text1> | <text2>' - Compare two texts")
        print("  • 'categories' - Show available sample categories")
        print("  • 'demo' - Run comprehensive demonstration")
        print("  • 'viz' - Create visualizations")
        print("  • 'quit' - Exit")
        print("-" * 50)
        
        # Pre-load sample embeddings
        print("🔄 Pre-loading sample embeddings...")
        embedded_categories = self.embed_categories(self.sample_texts)
        all_sample_texts = [text for texts in self.sample_texts.values() for text in texts]
        
        while True:
            try:
                user_input = input("\n🧠 Explorer: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n🧠 Thanks for exploring embeddings! 🚀")
                    break
                
                elif user_input.lower() == 'demo':
                    self.run_comprehensive_demo()
                
                elif user_input.lower() == 'viz':
                    try:
                        print("Creating visualizations...")
                        self.visualize_embeddings_2d(embedded_categories)
                        self.visualize_similarity_heatmap(embedded_categories)
                    except Exception as e:
                        print(f"❌ Visualization error: {e}")
                
                elif user_input.lower() == 'categories':
                    print("\n📂 Available Categories:")
                    for category, texts in self.sample_texts.items():
                        print(f"  • {category.title()}: {len(texts)} texts")
                        print(f"    Example: '{texts[0]}'")
                
                elif user_input.startswith('embed '):
                    text = user_input[6:].strip()
                    if text:
                        embedding = self.embed_text(text)
                        print(f"\n✅ Embedding created:")
                        print(f"  Text: '{text}'")
                        print(f"  Dimensions: {len(embedding.vector)}")
                        print(f"  Norm: {np.linalg.norm(embedding.vector):.3f}")
                        print(f"  Sample values: {embedding.vector[:5]}")
                    else:
                        print("❌ Please provide text after 'embed'")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print(f"\n🔍 Searching for: '{query}'")
                        results = self.find_most_similar(query, all_sample_texts, top_k=5)
                        
                        print("\n📊 Top Results:")
                        for rank, (text, similarity) in enumerate(results, 1):
                            # Find category
                            category = "unknown"
                            for cat, texts in self.sample_texts.items():
                                if text in texts:
                                    category = cat
                                    break
                            
                            print(f"  {rank}. [{category.upper()}] {text}")
                            print(f"     Similarity: {similarity:.3f}")
                    else:
                        print("❌ Please provide search query")
                
                elif ' | ' in user_input and user_input.startswith('compare '):
                    compare_text = user_input[8:].strip()
                    if ' | ' in compare_text:
                        text1, text2 = compare_text.split(' | ', 1)
                        text1, text2 = text1.strip(), text2.strip()
                        
                        if text1 and text2:
                            emb1 = self.embed_text(text1)
                            emb2 = self.embed_text(text2)
                            
                            cosine_sim = self.calculate_similarity(emb1.vector, emb2.vector, "cosine")
                            euclidean_sim = self.calculate_similarity(emb1.vector, emb2.vector, "euclidean")
                            dot_sim = self.calculate_similarity(emb1.vector, emb2.vector, "dot_product")
                            
                            print(f"\n📊 Similarity Analysis:")
                            print(f"  Text 1: '{text1}'")
                            print(f"  Text 2: '{text2}'")
                            print(f"  Cosine similarity: {cosine_sim:.3f}")
                            print(f"  Euclidean similarity: {euclidean_sim:.3f}")
                            print(f"  Dot product similarity: {dot_sim:.3f}")
                            
                            if cosine_sim > 0.7:
                                print("  🟢 Very similar texts")
                            elif cosine_sim > 0.5:
                                print("  🟡 Somewhat similar texts")
                            else:
                                print("  🔴 Different texts")
                        else:
                            print("❌ Please provide two texts separated by ' | '")
                    else:
                        print("❌ Use format: compare <text1> | <text2>")
                
                elif not user_input:
                    continue
                
                else:
                    print("❌ Unknown command. Type 'demo', 'embed <text>', 'search <query>', 'compare <text1> | <text2>', 'categories', 'viz', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy embedding!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def explain_embedding_concepts():
    """Explain key embedding concepts."""
    print("\n🧠 Embedding Vector Concepts")
    print("=" * 40)
    
    print("\n**What are Embeddings?**")
    print("• Mathematical representation of text/data as dense vectors")
    print("• Capture semantic meaning in numerical form")
    print("• Similar meanings → similar vectors in vector space")
    print("• Enable computers to understand and compare text")
    
    print("\n**Key Properties:**")
    print("• **Dimensionality**: Usually 50-4096 dimensions")
    print("• **Dense**: Every dimension has a meaningful value")
    print("• **Learned**: Trained on large text datasets")
    print("• **Contextual**: Modern models consider surrounding words")
    
    print("\n**Similarity Metrics:**")
    print("• **Cosine Similarity**: Measures angle between vectors (most common)")
    print("• **Euclidean Distance**: Geometric distance in vector space")
    print("• **Dot Product**: Direct multiplication of vector components")
    
    print("\n**Applications:**")
    print("• **Semantic Search**: Find similar documents/texts")
    print("• **RAG Systems**: Retrieve relevant context for LLMs")
    print("• **Recommendation**: Suggest similar items")
    print("• **Classification**: Group texts by similarity")
    
    print("\n**Popular Embedding Models:**")
    print("• **OpenAI text-embedding-ada-002**: General purpose, 1536 dims")
    print("• **Sentence-BERT**: Specialized for sentence similarity")
    print("• **Universal Sentence Encoder**: Google's multi-language model")
    print("• **BGE**: Beijing Academy of AI's high-performance embeddings")

def main():
    """Main function."""
    print("🚀 Starting Day 12: Embedding Vectors")
    print("Learning: Vector embeddings for semantic understanding")
    
    # Explain concepts first
    explain_embedding_concepts()
    
    try:
        # Create explorer and run
        explorer = EmbeddingExplorer()
        explorer.interactive_embedding_explorer()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install numpy matplotlib seaborn scikit-learn requests")
        print("\n🔄 Trying basic demo without visualizations...")
        
        try:
            explorer = EmbeddingExplorer()
            explorer.demonstrate_semantic_search()
        except Exception as e2:
            print(f"❌ Basic demo also failed: {e2}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()