from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os

def build_vector_db():
    # Verify numpy version
    print(f"Using numpy version: {np.__version__}")  # Should be 1.x
    
    # Define proper file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(current_dir, "../outputs/combined_texts.txt")
    
    # Check if file exists
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found at: {text_file}")
    
    # Load documents
    with open(text_file, "r", encoding="utf-8") as f:
        documents = f.read().splitlines()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = text_splitter.create_documents(documents)
    
    # Initialize model directly
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings
    embeddings = model.encode([chunk.page_content for chunk in chunks])
    
    # Create FAISS index
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip([chunk.page_content for chunk in chunks], embeddings)),
        embedding=model
    )
    
    # Create outputs directory if it doesn't exist
    output_dir = os.path.join(current_dir, "../outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    vector_db.save_local(os.path.join(output_dir, "resumes_vector_db"))

if __name__ == "__main__":
    build_vector_db()