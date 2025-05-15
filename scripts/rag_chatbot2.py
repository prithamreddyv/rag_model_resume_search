from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

def load_rag():
    """Load the RAG components with error handling"""
    try:
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Path to your existing FAISS index
        faiss_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../outputs/resumes_vector_db"
        )
        
        # Verify FAISS files exist
        if not all(os.path.exists(f"{faiss_path}/{f}") for f in ["index.faiss", "index.pkl"]):
            raise FileNotFoundError(
                f"FAISS files not found in {faiss_path}. "
                "Please ensure both index.faiss and index.pkl exist."
            )
        
        # Load FAISS vector DB
        db = FAISS.load_local(
            folder_path=faiss_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize Ollama
        llm = Ollama(
            model="llama3:8b-instruct-q4_0",
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        return db.as_retriever(search_kwargs={"k": 3}), llm
    
    except Exception as e:
        print(f"Error loading RAG components: {str(e)}")
        raise

def create_qa_chain():
    """Create the QA chain with the loaded components"""
    retriever, llm = load_rag()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def format_response(response):
    """Format the response to include IDs and source information"""
    result = response['result']
    sources = []
    
    for doc in response['source_documents']:
        # Extract ID from the document content (assuming format "ID:123 | RESUME:...")
        content = doc.page_content
        if "ID:" in content and "|" in content:
            doc_id = content.split("ID:")[1].split("|")[0].strip()
        else:
            doc_id = "N/A"
        
        # Extract first 100 characters of the content for preview
        preview = content[:100] + "..." if len(content) > 100 else content
        sources.append(f"ID: {doc_id}\nContent Preview: {preview}")
    
    formatted_output = f"Bot Response: {result}\n\nSource Documents:\n"
    formatted_output += "\n\n".join(f"• {source}" for source in sources)
    return formatted_output

def cli_chatbot():
    """Run the interactive chatbot interface"""
    try:
        qa = create_qa_chain()
        print("Chatbot ready! Type 'exit' to quit.\n")
        print("Note: All responses will include resume IDs for verification.\n")
        
        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                response = qa.invoke({"query": query})
                print(f"\n{format_response(response)}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")

if __name__ == "__main__":
    cli_chatbot()