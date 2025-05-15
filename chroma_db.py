import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

# Load dataset
resume_df = pd.read_csv("../data/synthetic-resumes.csv")

# Convert DataFrame to LangChain documents
loader = DataFrameLoader(resume_df, page_content_column="Resume")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Use Ollama for embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store in ChromaDB locally
vector_db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")

print("Vector database created successfully! Upload 'chroma_db' folder to Google Colab.")
