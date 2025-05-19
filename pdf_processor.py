import os
from pathlib import Path
from typing import List, Optional, Dict
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from dotenv import load_dotenv
import hashlib
import json
import shutil
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path("DATA")
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_FILE = DATA_DIR / "pdf_metadata.json"

# Add cache configuration
CACHE_SIZE = 100  # Number of queries to cache
QUERY_CACHE_FILE = DATA_DIR / "query_cache.json"

def get_pdf_hash(pdf_path: Path) -> str:
    """Generate a hash for the PDF file to track changes."""
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_metadata() -> dict:
    """Load PDF metadata from JSON file."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                content = f.read().strip()
                if content:  # Check if file is not empty
                    return json.loads(content)
                else:
                    print(f"Warning: {METADATA_FILE} exists but is empty. Returning empty dict.")
                    return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {METADATA_FILE}: {e}. Creating new metadata.")
            # Backup the corrupted file
            backup_file = METADATA_FILE.with_suffix('.json.bak')
            shutil.copy2(METADATA_FILE, backup_file)
            print(f"Backed up corrupted metadata to {backup_file}")
            return {}
    return {}

def save_metadata(metadata: dict):
    """Save PDF metadata to JSON file."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def process_pdf(pdf_path: Path) -> VectorStoreIndex:
    """Process a single PDF file and create its vector store."""
    print(f"Processing PDF: {pdf_path}")
    
    # Create embeddings directory
    embedding_dir = EMBEDDINGS_DIR / pdf_path.stem
    if embedding_dir.exists():
        shutil.rmtree(embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the PDF
    documents = SimpleDirectoryReader(
        input_files=[str(pdf_path)],
        recursive=True,
        filename_as_id=True
    ).load_data()
    
    # Configure node parser for better text chunking
    node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separator="\n"
    )
    
    # Configure settings
    Settings.embed_model = OpenAIEmbedding()
    Settings.node_parser = node_parser
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    # Save embeddings
    index.storage_context.persist(str(embedding_dir))
    print(f"Embeddings saved to: {embedding_dir}")
    
    return index

def process_new_pdfs() -> List[str]:
    """Process any new PDFs in the raw_pdfs directory."""
    metadata = load_metadata()
    processed_files = []
    
    # Create directories if they don't exist
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each PDF
    pdf_files = list(RAW_PDFS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {RAW_PDFS_DIR}")
    
    for pdf_path in pdf_files:
        pdf_hash = get_pdf_hash(pdf_path)
        
        # Skip if PDF hasn't changed
        if pdf_path.name in metadata and metadata[pdf_path.name]['hash'] == pdf_hash:
            print(f"Skipping {pdf_path.name} - already processed and unchanged")
            continue
        
        print(f"Processing new or changed PDF: {pdf_path.name}")
        # Process PDF
        process_pdf(pdf_path)
        
        # Update metadata
        metadata[pdf_path.name] = {
            'hash': pdf_hash,
            'processed_date': str(pdf_path.stat().st_mtime),
            'embedding_dir': str(EMBEDDINGS_DIR / pdf_path.stem)
        }
        processed_files.append(pdf_path.name)
    
    if processed_files:
        save_metadata(metadata)
        print(f"Processed {len(processed_files)} new PDFs: {processed_files}")
    else:
        print("No new PDFs to process")
    
    return processed_files

def get_vector_store(pdf_name: Optional[str] = None) -> VectorStoreIndex:
    """Get vector store for a specific PDF or all PDFs."""
    metadata = load_metadata()
    
    if pdf_name:
        if pdf_name not in metadata:
            raise ValueError(f"No embeddings found for {pdf_name}")
        embedding_dir = Path(metadata[pdf_name]['embedding_dir'])
        if not embedding_dir.exists():
            raise ValueError(f"Embedding directory not found for {pdf_name}")
        storage_context = StorageContext.from_defaults(persist_dir=str(embedding_dir))
        return load_index_from_storage(storage_context)
    
    # If no specific PDF is requested, combine all embeddings
    indices = []
    for pdf_info in metadata.values():
        embedding_dir = Path(pdf_info['embedding_dir'])
        if embedding_dir.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(embedding_dir))
            indices.append(load_index_from_storage(storage_context))
    
    if not indices:
        raise ValueError("No embeddings found for any PDFs")
    
    return indices[0] if len(indices) == 1 else indices

def load_query_cache() -> Dict:
    """Load query cache from file."""
    if QUERY_CACHE_FILE.exists():
        with open(QUERY_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_query_cache(cache: Dict):
    """Save query cache to file."""
    with open(QUERY_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_query(query: str, pdf_name: Optional[str] = None) -> Optional[str]:
    """Get cached query result if available."""
    cache = load_query_cache()
    cache_key = f"{pdf_name}:{query}" if pdf_name else query
    return cache.get(cache_key)

def cache_query_result(query: str, result: str, pdf_name: Optional[str] = None):
    """Cache query result."""
    cache = load_query_cache()
    cache_key = f"{pdf_name}:{query}" if pdf_name else query
    cache[cache_key] = result
    save_query_cache(cache)

def query_pdf(query: str, pdf_name: Optional[str] = None) -> str:
    """Query the PDF(s) using the vector store with enhanced retrieval."""
    try:
        # Add debugging to check available PDFs
        metadata = load_metadata()
        available_pdfs = list(metadata.keys())
        print(f"Available PDFs in metadata: {available_pdfs}")
        
        # Get the vector store
        index_or_indices = get_vector_store(pdf_name)
        
        # Handle the case when multiple indices are returned
        if isinstance(index_or_indices, list):
            # Use the first index for simplicity or implement a more complex solution
            print(f"Multiple indices found. Using the first one from: {pdf_name if pdf_name else 'all PDFs'}")
            index = index_or_indices[0]
        else:
            index = index_or_indices
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,  # Retrieve top 5 most relevant chunks
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT  # Use default query mode
        )
        
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer()
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        
        # Execute query
        response = query_engine.query(query)
        
        # Format response with sources
        if hasattr(response, 'source_nodes'):
            sources = [node.node.text for node in response.source_nodes]
            return f"Based on the document:\n\n{response.response}\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
        return str(response.response)
        
    except Exception as e:
        print(f"Error querying PDF: {str(e)}")
        return f"Error querying PDF: {str(e)}"

# Initialize processing of any new PDFs when this module is imported
if __name__ == "__main__":
    print("Running PDF processor directly")
    processed = process_new_pdfs()
    print(f"Processed files: {processed}")
else:
    print("PDF processor module loaded") 