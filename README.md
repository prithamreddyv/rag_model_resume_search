RAG Resume Retrieval System
===========================

A Retrieval-Augmented Generation (RAG) chatbot that semantically searches and retrieves resumes based on natural language queries. It uses a MiniLM (BERT-based) model for embeddings, FAISS for vector storage, LangChain for chaining logic, and Ollama for running a local LLM.

-----------------------------------------
FEATURES
-----------------------------------------
- Semantic resume search using vector similarity
- Local FAISS index for fast retrieval
- Uses all-MiniLM-L6-v2 (a BERT variant) for embedding
- LLM powered by Ollama (e.g., llama3)
- Chat memory with ConversationalRetrievalChain
- CLI and Gradio-based web UI support
- Resume ID selection and full resume display

-----------------------------------------
PROJECT STRUCTURE
-----------------------------------------
.
├── data/
│   └── synthetic-resumes.csv
├── outputs2/
│   ├── combined_texts.txt
│   └── resumes_vector_db/
│       ├── index.faiss
│       └── index.pkl
├── build_vectordb.py
├── rag_chatbot.py
├── requirements.txt
└── README.txt

-----------------------------------------
SETUP INSTRUCTIONS
-----------------------------------------

1. Clone the repository:
   git clone https://github.com/prithamreddyv/rag-resume-retriever.git
   cd rag-resume-retriever

2. Install Python requirements:
   pip install -r requirements.txt

   Example packages:
   - langchain
   - langchain_community
   - langchain_huggingface
   - sentence-transformers
   - faiss-cpu
   - pandas
   - gradio

3. Start Ollama (for local LLM):
   - Install from https://ollama.com
   - Run: ollama pull llama3
   - Ensure it’s running at http://localhost:11434

-----------------------------------------
BUILDING THE VECTOR DATABASE
-----------------------------------------
- Make sure `outputs2/combined_texts.txt` exists
- Run the script to embed and store resume data:

   python build_vectordb.py

   This creates:
   - Dense vectors from resumes
   - A FAISS index in outputs2/resumes_vector_db

-----------------------------------------
RUNNING THE CHATBOT
-----------------------------------------

Option 1: Command-line interface
   python rag_chatbot.py
   → Choose option 1

Option 2: Gradio web interface
   python rag_chatbot.py
   → Choose option 2
   → Opens in browser

-----------------------------------------
SAMPLE QUERIES
-----------------------------------------
- "Python developer with Azure"
- "Who knows computer vision and PyTorch?"
- "Show resumes for data engineer with AWS"

-----------------------------------------
NOTES
-----------------------------------------
- Embeddings are generated using all-MiniLM-L6-v2
- LangChain handles memory, chains, and LLM interaction
- FAISS provides fast vector similarity search
- Ollama powers the local language model (e.g., llama3)

-----------------------------------------
LICENSE
-----------------------------------------
MIT License

-----------------------------------------
CREDITS / TOOLS USED
-----------------------------------------
- SentenceTransformers (MiniLM / BERT)
- FAISS (Facebook AI Similarity Search)
- LangChain
- Ollama
- Gradio
