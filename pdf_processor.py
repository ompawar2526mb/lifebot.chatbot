import os
from typing import List, Dict
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm
import logging
import json
from pathlib import Path
import hashlib
import shutil
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class PDFProcessor:
    def __init__(self, data_dir: str = "DATA"):
        """Initialize the PDF processor with FAISS vector store and OpenAI embeddings."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize directory structure
        self.data_dir = Path(data_dir)
        self.raw_pdfs_dir = self.data_dir / "raw_pdfs"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        # Create directories if they don't exist
        self.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.index = None  # Will be initialized when we have data
        
        # Document store
        self.documents: List[Dict] = []
        
        # Configure chunking
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using NLTK to remove low-weighted words while preserving semantic meaning.
        
        Args:
            text: Input text (document chunk or query)
            
        Returns:
            str: Preprocessed text with high-weighted words
        """
        try:
            # Tokenize the text
            tokens = word_tokenize(text.lower())
            
            # Remove stop words and non-alphabetic tokens, apply lemmatization
            filtered_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words
            ]
            
            # Reconstruct the text
            processed_text = " ".join(filtered_tokens)
            return processed_text if processed_text else text  # Fallback to original if empty
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text  # Fallback to original text on error
    
    def process_pdfs(self) -> None:
        """Process all PDFs in the raw_pdfs directory."""
        if not self.raw_pdfs_dir.exists():
            logger.error(f"Raw PDFs directory not found: {self.raw_pdfs_dir}")
            return
        
        pdf_files = [f for f in self.raw_pdfs_dir.glob("*.pdf") if f.name != ".gitkeep"]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_pdfs_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {pdf_path.name}")
            logger.info(f"{'='*50}")
            
            try:
                self.process_pdf(str(pdf_path))
                logger.info(f"Successfully processed {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        """Generate a hash for the PDF file to use as a unique identifier."""
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_pdf_storage_path(self, pdf_path: str) -> Path:
        """Get the storage path for a specific PDF."""
        pdf_hash = self._get_pdf_hash(pdf_path)
        pdf_name = Path(pdf_path).stem
        pdf_dir = self.embeddings_dir / f"{pdf_name}_{pdf_hash}"
        pdf_dir.mkdir(exist_ok=True)
        return pdf_dir
    
    def _copy_pdf_to_raw(self, pdf_path: str) -> Path:
        """Copy PDF to raw_pdfs directory and return the new path."""
        pdf_name = Path(pdf_path).name
        target_path = self.raw_pdfs_dir / pdf_name
        
        # Copy the file if it's not already in raw_pdfs
        if not target_path.exists():
            shutil.copy2(pdf_path, target_path)
            logger.info(f"Copied {pdf_path} to {target_path}")
        
        return target_path
    
    def _load_pdf_data(self, pdf_path: str) -> tuple:
        """Load existing data for a specific PDF if available."""
        pdf_dir = self._get_pdf_storage_path(pdf_path)
        index_path = pdf_dir / "faiss_index.bin"
        documents_path = pdf_dir / "documents.json"
        
        if index_path.exists() and documents_path.exists():
            logger.info(f"Loading existing data for {pdf_path}")
            index = faiss.read_index(str(index_path))
            
            with open(documents_path, 'r') as f:
                documents = json.load(f)
                # Convert string embeddings back to numpy arrays
                for doc in documents:
                    doc['embedding'] = np.array(doc['embedding'])
            
            return index, documents
        
        return None, []
    
    def _save_pdf_data(self, pdf_path: str, index: faiss.Index, documents: List[Dict]):
        """Save data for a specific PDF."""
        pdf_dir = self._get_pdf_storage_path(pdf_path)
        
        # Save FAISS index
        index_path = pdf_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        
        # Save documents
        documents_path = pdf_dir / "documents.json"
        # Convert numpy arrays to lists for JSON serialization
        documents_to_save = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_copy['embedding'] = doc_copy['embedding'].tolist()
            documents_to_save.append(doc_copy)
        
        with open(documents_path, 'w') as f:
            json.dump(documents_to_save, f)
    
    def _initialize_index(self, n_vectors: int):
        """Initialize FAISS index based on the number of vectors."""
        if self.index is not None:
            return
            
        # For small datasets, use a simple flat index
        if n_vectors < 100:
            logger.info("Using flat index for small dataset")
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # For larger datasets, use IVF index
            nlist = min(n_vectors // 10, 100)  # Number of clusters
            logger.info(f"Using IVF index with {nlist} clusters")
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.nprobe = min(nlist // 10, 10)  # Number of clusters to search
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Process a PDF file and return its chunks with embeddings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Copy PDF to raw_pdfs directory
        raw_pdf_path = self._copy_pdf_to_raw(pdf_path)
        
        # Check if we already have processed this PDF
        existing_index, existing_documents = self._load_pdf_data(raw_pdf_path)
        if existing_index is not None:
            logger.info(f"Using cached embeddings for {pdf_path}")
            self.index = existing_index
            self.documents = existing_documents
            return self.documents
        
        # Load and parse PDF using LlamaIndex with optimized chunking
        reader = SimpleDirectoryReader(input_files=[str(raw_pdf_path)])
        documents = reader.load_data()
        
        # Use SentenceSplitter for better semantic chunking
        parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        nodes = parser.get_nodes_from_documents(documents)
        
        logger.info(f"Created {len(nodes)} chunks from the document")
        
        # Extract text chunks
        original_chunks = [node.text for node in nodes]
        
        # Apply NLTK preprocessing to reduce token count
        logger.info("Applying NLTK preprocessing to reduce token count...")
        preprocessed_chunks = []
        total_original_tokens = 0
        total_processed_tokens = 0
        
        for chunk in original_chunks:
            # Preprocess the text to reduce tokens
            processed_chunk = self.preprocess_text(chunk)
            preprocessed_chunks.append(processed_chunk)
            
            # Calculate token reduction statistics (simple token count)
            original_tokens = len(word_tokenize(chunk))
            processed_tokens = len(word_tokenize(processed_chunk))
            total_original_tokens += original_tokens
            total_processed_tokens += processed_tokens
        
        # Log token reduction statistics
        reduction_percent = ((total_original_tokens - total_processed_tokens) / total_original_tokens) * 100 if total_original_tokens > 0 else 0
        logger.info(f"Token reduction: {total_original_tokens} → {total_processed_tokens} tokens ({reduction_percent:.2f}% reduction)")
        
        # Generate embeddings using OpenAI with preprocessed chunks
        logger.info("Generating embeddings...")
        embeddings = self._get_embeddings_batch(preprocessed_chunks)
        
        # Initialize or update FAISS index
        self._initialize_index(len(embeddings))
        
        # Add to FAISS index
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
        
        logger.info("Adding vectors to FAISS index...")
        self.index.add(embeddings)
        
        # Store documents with metadata
        self.documents = [
            {
                "text": processed_chunk,  # Store preprocessed text
                "embedding": embedding,
                "metadata": {
                    "source": str(raw_pdf_path),
                    "chunk_id": i,
                    "chunk_size": len(processed_chunk),
                    "original_text": original_chunk  # Store original text for reference if needed
                }
            }
            for i, (original_chunk, processed_chunk, embedding) in enumerate(zip(original_chunks, preprocessed_chunks, embeddings))
        ]
        
        # Save the data for this PDF
        self._save_pdf_data(raw_pdf_path, self.index, self.documents)
        
        logger.info(f"Successfully processed {len(preprocessed_chunks)} chunks from {pdf_path}")
        return self.documents
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if not self.documents:
            return []
        
        # Note: Query is already preprocessed in rag.py, but we log the effect for consistency
        logger.info("Query received (already preprocessed in RAGSystem): applying minimal preprocessing for logging...")
        original_query = query
        processed_query = self.preprocess_text(query)
        
        # Log token reduction for the query (for informational purposes)
        original_tokens = len(word_tokenize(original_query))
        processed_tokens = len(word_tokenize(processed_query))
        reduction_percent = ((original_tokens - processed_tokens) / original_tokens) * 100 if original_tokens > 0 else 0
        logger.info(f"Query token reduction (informational): {original_tokens} → {processed_tokens} tokens ({reduction_percent:.2f}% reduction)")
        
        # Generate query embedding using OpenAI with the preprocessed query
        query_embedding = self._get_embedding(query)  # Use the already preprocessed query from rag.py
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Return results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx]["text"],
                    "score": float(1 - distance),  # Convert distance to similarity score
                    "metadata": self.documents[idx]["metadata"]
                })
        
        return results

def main():
    """Run the PDF processor independently."""
    processor = PDFProcessor()
    processor.process_pdfs()

if __name__ == "__main__":
    main()