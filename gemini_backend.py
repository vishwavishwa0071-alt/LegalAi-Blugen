import os
import json
import logging
import asyncio
import numpy as np
import pickle
from typing import List
from dotenv import load_dotenv
import streamlit as st
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"
METADATA_FILE = "cpc_metadata.json"
GEMINI_INDEX_FILE = "gemini_embeddings.pkl"

# Import prompts
try:
    from prompts import response_prompt
except ImportError:
    # Fallback prompt if prompts.py doesn't exist
    response_prompt = """
    Based on the following context from the legal document, answer the user's question.
    
    Context:
    {context_chunks}
    
    User Question: {user_question}
    
    Provide a detailed, accurate answer based only on the information in the context above.
    """
    logging.warning("Using fallback prompt. prompts.py not found.")


class GeminiRAGBackend:
    """RAG Backend using Gemini embeddings and Gemini-2.5-flash-lite LLM"""
    
    def __init__(self):
        """Initialize the Gemini RAG backend"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets or environment variables")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Load chunks
        self.chunks = self._load_chunks()
        logging.info(f"Loaded {len(self.chunks)} chunks from metadata")
        
        # Load or create embeddings
        self.chunk_embeddings = self._load_or_create_embeddings()
        logging.info("Embeddings ready")
    
    def _load_chunks(self) -> List[str]:
        """Load chunks from metadata file"""
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                f"Metadata file {METADATA_FILE} not found. "
                "Run embedding_comparison.py first to generate chunks."
            )
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        return chunks
    
    def _load_or_create_embeddings(self) -> np.ndarray:
        """Load existing embeddings or create new ones using Gemini"""
        if os.path.exists(GEMINI_INDEX_FILE):
            logging.info(f"Loading existing embeddings from {GEMINI_INDEX_FILE}")
            with open(GEMINI_INDEX_FILE, 'rb') as f:
                embeddings = pickle.load(f)
            return embeddings
        
        logging.info("Creating new embeddings using Gemini API...")
        return self._create_embeddings()
    
    def _create_embeddings(self) -> np.ndarray:
        """Create embeddings for all chunks using Gemini API"""
        logging.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        
        # Gemini API can handle batch embedding
        # Process in batches to avoid API limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size}")
            
            try:
                result = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch_chunks
                )
                
                # Extract embedding values
                batch_embeddings = [emb.values for emb in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logging.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Save embeddings for future use
        with open(GEMINI_INDEX_FILE, 'wb') as f:
            pickle.dump(embeddings_array, f)
        
        logging.info(f"Saved embeddings to {GEMINI_INDEX_FILE}")
        return embeddings_array
    
    def _cosine_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all chunks"""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunks_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(chunks_norm, query_norm)
        return similarities
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """Search for most relevant chunks using Gemini embeddings"""
        logging.info(f"Searching for top {k} relevant chunks...")
        
        # Get query embedding
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query]
        )
        query_embedding = np.array(result.embeddings[0].values, dtype=np.float32)
        
        # Calculate similarities
        similarities = self._cosine_similarity(query_embedding, self.chunk_embeddings)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top k chunks
        retrieved_chunks = [self.chunks[idx] for idx in top_k_indices]
        
        # Log similarity scores
        for i, idx in enumerate(top_k_indices):
            logging.info(f"  Rank {i+1}: Chunk {idx}, Similarity: {similarities[idx]:.4f}")
        
        return retrieved_chunks
    
    def generate_response(self, context_chunks: List[str], user_question: str) -> str:
        """Generate response using Gemini-2.5-flash-lite"""
        logging.info("Generating response using Gemini LLM...")
        
        # Combine context chunks
        context = "\n\n---\n\n".join(context_chunks)
        
        # Format prompt
        prompt = response_prompt.format(
            context_chunks=context,
            user_question=user_question
        )
        
        try:
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def get_answer(self, user_question: str) -> str:
        """
        Main method to get answer for a user question
        1. Search for relevant chunks
        2. Generate response using Gemini LLM
        """
        logging.info(f"Processing query: {user_question}")
        
        # Search for relevant chunks
        retrieved_chunks = self.search(user_question, k=4)
        
        # Generate response
        response = self.generate_response(retrieved_chunks, user_question)
        
        logging.info("Response generated successfully")
        return response


class GeminiRAGComparison:
    """Compare RAG performance with different configurations"""
    
    def __init__(self):
        self.backend = GeminiRAGBackend()
    
    async def test_with_different_k(self, question: str, k_values: List[int] = [2, 4, 6, 8]):
        """Test RAG performance with different number of retrieved chunks"""
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing question: {question}")
        logging.info(f"{'='*80}\n")
        
        results = {}
        
        for k in k_values:
            logging.info(f"\n--- Testing with k={k} ---")
            chunks = self.backend.search(question, k=k)
            response = self.backend.generate_response(chunks, question)
            
            results[k] = {
                "chunks": chunks,
                "response": response,
                "num_chunks": len(chunks)
            }
            
            print(f"\n{'='*80}")
            print(f"RESULTS WITH k={k}")
            print(f"{'='*80}")
            print(f"\nResponse:\n{response}\n")
        
        return results


if __name__ == "__main__":
    async def main():
        print("="*80)
        print("Gemini RAG Backend - Testing")
        print("="*80)
        
        try:
            # Initialize backend
            print("\n[1/3] Initializing Gemini RAG Backend...")
            backend = GeminiRAGBackend()
            print("✓ Backend initialized successfully\n")
            
            # Test questions
            test_questions = [
                "What is the procedure for filing a suit?",
                "What are the rules for service of summons?",
                "Explain the concept of res judicata."
            ]
            
            # Interactive mode
            print("\n[2/3] Available test modes:")
            print("  1. Test with predefined questions")
            print("  2. Interactive Q&A")
            print("  3. Compare different k values")
            
            mode = input("\nSelect mode (1/2/3): ").strip()
            
            if mode == "1":
                print("\n[3/3] Testing with predefined questions...\n")
                for i, question in enumerate(test_questions, 1):
                    print(f"\n{'='*80}")
                    print(f"Question {i}/{len(test_questions)}: {question}")
                    print(f"{'='*80}")
                    
                    answer = await backend.get_answer(question)
                    
                    print(f"\nAnswer:\n{answer}\n")
                    print("-"*80)
            
            elif mode == "2":
                print("\n[3/3] Interactive Q&A Mode (type 'quit' to exit)...\n")
                while True:
                    question = input("\nYour question: ").strip()
                    if question.lower() in ['quit', 'q', 'exit']:
                        break
                    
                    if not question:
                        continue
                    
                    print("\nProcessing...")
                    answer = await backend.get_answer(question)
                    print(f"\nAnswer:\n{answer}\n")
                    print("-"*80)
            
            elif mode == "3":
                print("\n[3/3] Comparing different k values...\n")
                question = input("Enter question to test: ").strip()
                if question:
                    comparison = GeminiRAGComparison()
                    await comparison.test_with_different_k(question)
            
            else:
                print("Invalid mode selected")
            
            print("\n✓ Testing completed!")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            logging.error(f"Fatal error: {e}", exc_info=True)
    
    # Run the async main function
    asyncio.run(main())
