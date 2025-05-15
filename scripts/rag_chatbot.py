from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import gradio as gr

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

def cli_chatbot():
    """Run the interactive chatbot interface"""
    try:
        qa = create_qa_chain()
        print("Chatbot ready! Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                result = qa.invoke({"query": query})
                print(f"\nBot: {result['result']}")
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")


if __name__ == "__main__":
    cli_chatbot()



# from langchain_community.llms import Ollama
# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# import os
# import gradio as gr

# def load_rag():
#     """Load the RAG components with error handling"""
#     try:
#         # Load embeddings model
#         embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
        
#         # Path to your existing FAISS index
#         faiss_path = os.path.join(
#             os.path.dirname(os.path.abspath(__file__)),
#             "../outputs/resumes_vector_db"
#         )
        
#         # Verify FAISS files exist
#         if not all(os.path.exists(f"{faiss_path}/{f}") for f in ["index.faiss", "index.pkl"]):
#             raise FileNotFoundError(
#                 f"FAISS files not found in {faiss_path}. "
#                 "Please ensure both index.faiss and index.pkl exist."
#             )
        
#         # Load FAISS vector DB
#         db = FAISS.load_local(
#             folder_path=faiss_path,
#             embeddings=embeddings,
#             allow_dangerous_deserialization=True
#         )
        
#         # Initialize Ollama
#         llm = Ollama(
#             model="llama3:8b-instruct-q4_0",
#             base_url="http://localhost:11434",
#             temperature=0.3
#         )
        
#         return db.as_retriever(search_kwargs={"k": 3}), llm
    
#     except Exception as e:
#         print(f"Error loading RAG components: {str(e)}")
#         raise

# def create_qa_chain():
#     """Create the QA chain with the loaded components"""
#     retriever, llm = load_rag()
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

# def cli_chatbot():
#     """Run the interactive chatbot interface"""
#     try:
#         qa = create_qa_chain()
#         print("Chatbot ready! Type 'exit' to quit.\n")
        
#         while True:
#             try:
#                 query = input("You: ")
#                 if query.lower() in ["exit", "quit"]:
#                     break
                
#                 result = qa.invoke({"query": query})
#                 print(f"\nBot: {result['result']}")
#                 print()
                
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break
#             except Exception as e:
#                 print(f"Error processing query: {str(e)}")
                
#     except Exception as e:
#         print(f"Failed to initialize chatbot: {str(e)}")

# def gradio_chatbot(message, history):
#     """Gradio chatbot interface function"""
#     try:
#         # Create QA chain if not already created
#         qa = create_qa_chain()
        
#         # Process the query
#         result = qa.invoke({"query": message})
        
#         # Return the response
#         return result['result']
    
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# def launch_gradio_ui():
#     """Launch the Gradio UI for the chatbot"""
#     demo = gr.ChatInterface(
#         fn=gradio_chatbot,
#         title="Resume RAG Chatbot",
#         description="Ask questions about the resume corpus",
#         theme="soft"
#     )
    
#     demo.launch(
#         share=False,  # Set to True if you want a public link
#         # server_name="0.0.0.0",  # Allows access from other devices on the network
#         server_port=7860  # Specify a port
#     )

# if __name__ == "__main__":
#     # Option to choose between CLI and Gradio UI
#     interface_choice = input("Choose interface (1 for CLI, 2 for Gradio UI): ").strip()
    
#     if interface_choice == "1":
#         cli_chatbot()
#     elif interface_choice == "2":
#         launch_gradio_ui()
#     else:
#         print("Invalid choice. Defaulting to CLI.")
#         cli_chatbot()