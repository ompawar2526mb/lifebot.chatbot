import os
from pathlib import Path
from typing import List, Optional
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

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path("DATA")
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_FILE = DATA_DIR / "pdf_metadata.json"

def get_pdf_hash(pdf_path: Path) -> str:
    """Generate a hash for the PDF file to track changes."""
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_metadata() -> dict:
    """Load PDF metadata from JSON file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata: dict):
    """Save PDF metadata to JSON file."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def process_pdf(pdf_path: Path) -> VectorStoreIndex:
    """Process a single PDF file and create its vector store."""
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
        show_progress=True,
        service_context=Settings.get_service_context()
    )
    
    # Save embeddings
    index.storage_context.persist(str(embedding_dir))
    
    return index

def process_new_pdfs() -> List[str]:
    """Process any new PDFs in the raw_pdfs directory."""
    metadata = load_metadata()
    processed_files = []
    
    # Create directories if they don't exist
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each PDF
    for pdf_path in RAW_PDFS_DIR.glob("*.pdf"):
        pdf_hash = get_pdf_hash(pdf_path)
        
        # Skip if PDF hasn't changed
        if pdf_path.name in metadata and metadata[pdf_path.name]['hash'] == pdf_hash:
            continue
        
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

def query_pdf(query: str, pdf_name: Optional[str] = None) -> str:
    """Query the PDF(s) using the vector store with enhanced retrieval."""
    try:
        # Get the vector store
        index = get_vector_store(pdf_name)
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,  # Retrieve top 5 most relevant chunks
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT  # Use default query mode
        )
        
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            structured_answer_filtering=True
        )
        
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

# Initialize processing of any new PDFs
process_new_pdfs() 