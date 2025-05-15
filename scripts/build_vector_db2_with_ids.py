# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# import numpy as np
# import os
# import re

# def build_vector_db():
#     print(f"Using numpy version: {np.__version__}")

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     text_file = os.path.join(current_dir, "../outputs2/combined_texts.txt")

#     if not os.path.exists(text_file):
#         raise FileNotFoundError(f"Text file not found at: {text_file}")

#     with open(text_file, "r", encoding="utf-8") as f:
#         raw_lines = f.read().splitlines()

#     print(f"Loaded {len(raw_lines)} resumes")

#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=512,
#         chunk_overlap=50
#     )

#     documents = []
#     for line in raw_lines:
#         # Use regex to extract resume_id
#         match = re.search(r'ID:(\d+)', line)
#         resume_id = match.group(1) if match else "unknown"

#         # Split each resume into smaller chunks
#         split_docs = text_splitter.create_documents([line])
#         for chunk in split_docs:
#             # Attach metadata
#             doc = Document(
#                 page_content=chunk.page_content,
#                 metadata={"resume_id": resume_id}
#             )
#             documents.append(doc)

#     print(f"Total chunks created: {len(documents)}")

#     # Load embedding model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Get embeddings for all chunks
#     embeddings = model.encode([doc.page_content for doc in documents])

#     # Build FAISS index using metadata
#     vector_db = FAISS.from_embeddings(
#         text_embeddings=list(zip([doc.page_content for doc in documents], embeddings)),
#         embedding=model,
#         metadatas=[doc.metadata for doc in documents]
#     )

#     output_dir = os.path.join(current_dir, "../outputs2")
#     os.makedirs(output_dir, exist_ok=True)

#     vector_db.save_local(os.path.join(output_dir, "resumes_vector_db"))
#     print("Vector DB saved successfully.")

# if __name__ == "__main__":
#     build_vector_db()








from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np
import os
import re

def build_vector_db():
    print(f"Using numpy version: {np.__version__}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(current_dir, "../outputs2/combined_texts.txt")

    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found at: {text_file}")

    with open(text_file, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    print(f"Loaded {len(raw_lines)} resumes")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    documents = []
    for line in raw_lines:
        # Extract resume_id using regex
        match = re.search(r'ID:(\d+)', line)
        resume_id = match.group(1) if match else "unknown"

        # Split each resume into chunks
        split_docs = text_splitter.create_documents([line])
        for chunk in split_docs:
            doc = Document(
                page_content=chunk.page_content,
                metadata={"resume_id": resume_id}
            )
            documents.append(doc)

    print(f"Total chunks created: {len(documents)}")

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings for all documents
    embeddings = model.encode([doc.page_content for doc in documents])

    # Build FAISS index using embeddings and metadata
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip([doc.page_content for doc in documents], embeddings)),
        embedding=model,
        metadatas=[doc.metadata for doc in documents]
    )

    output_dir = os.path.join(current_dir, "../outputs2")
    os.makedirs(output_dir, exist_ok=True)

    vector_db.save_local(os.path.join(output_dir, "resumes_vector_db"))
    print("Vector DB saved successfully.")

if __name__ == "__main__":
    build_vector_db()
