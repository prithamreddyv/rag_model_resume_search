# from langchain_community.llms import Ollama
# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
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
        
#         return db.as_retriever(search_type="similarity", search_kwargs={"k": 3}), llm
    
#     except Exception as e:
#         print(f"Error loading RAG components: {str(e)}")
#         raise


# def create_qa_chain():
#     """Create the QA chain using ConversationalRetrievalChain"""
#     retriever, llm = load_rag()

#     # Print test retrieval for debugging
#     test_docs = retriever.get_relevant_documents("test resume")
#     print("[DEBUG] Retriever Test Output:")
#     for doc in test_docs:
#         print(f"- Content Preview: {doc.page_content[:100]}")
#         print(f"  Metadata: {doc.metadata}\n")

#     # Optional: Maintain conversational history
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", return_messages=True
#     )

#     # Build ConversationalRetrievalChain
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=True
#     )

#     return qa_chain



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

#                 # Bot response
#                 print(f"\nBot: {result['result']}")

#                 # Extract and print resume IDs from metadata
#                 source_docs = result.get("source_documents", [])
#                 print("\nMatched Resume IDs:")
#                 found = False

#                 for doc in source_docs:
#                     resume_id = doc.metadata.get("resume_id", "unknown")
#                     if resume_id != "unknown":
#                         print(f"- Resume ID: {resume_id}")
#                         found = True

#                 if not found:
#                     print("- None found or metadata missing")

#                 # Debug: print full document metadata and preview
#                 print("\n[DEBUG] Source Documents Raw Output:")
#                 for i, doc in enumerate(source_docs, 1):
#                     print(f"--- Document {i} ---")
#                     print(f"Content Preview: {doc.page_content[:200]}...")
#                     print(f"Metadata: {doc.metadata}")

#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break
#             except Exception as e:
#                 print(f"Error processing query: {str(e)}")
                
#     except Exception as e:
#         print(f"Failed to initialize chatbot: {str(e)}")



# if __name__ == "__main__":
#     cli_chatbot()









from langchain_ollama import OllamaLLM  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import gradio as gr
import pandas as pd


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
            "../outputs2/resumes_vector_db"
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

        # Initialize OllamaLLM (new import path)
        llm = OllamaLLM(
            model="llama3:8b-instruct-q4_0",
            base_url="http://localhost:11434",
            temperature=0.3
        )

        return db.as_retriever(search_type="similarity", search_kwargs={"k": 3}), llm

    except Exception as e:
        print(f"Error loading RAG components: {str(e)}")
        raise


def create_qa_chain():
    """Create the QA chain using ConversationalRetrievalChain"""
    retriever, llm = load_rag()

    # Print test retrieval for debugging
    test_docs = retriever.get_relevant_documents("test resume")
    print("[DEBUG] Retriever Test Output:")
    for doc in test_docs:
        print(f"- Content Preview: {doc.page_content[:100]}")
        print(f"  Metadata: {doc.metadata}\n")

    # Maintain conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # 🔥 this line is crucial
    )

    # Build ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # ✅ Fixes multiple output key error
    )

    return qa_chain


def get_resume_by_id(resume_id):
    csv_path = os.path.join(os.path.dirname(__file__), "../outputs2/synthetic-resumes.csv")
    df = pd.read_csv(csv_path)

    # Make sure 'ID' column is treated as int
    df['ID'] = df['ID'].astype(str)

    # Look up resume by id
    resume_row = df[df['ID'] == str(resume_id)]
    if not resume_row.empty:
        return resume_row.iloc[0]['Resume']
    return f"Resume ID {resume_id} not found in dataset."




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

                result = qa.invoke({"question": query})  # key changed from 'query' to 'question'

                print(f"\nBot: {result['answer']}")

                source_docs = result.get("source_documents", [])
                print("\nMatched Resume IDs:")
                found = False
                for doc in source_docs:
                    resume_id = doc.metadata.get("resume_id", "unknown")
                    if resume_id != "unknown":
                        print(f"- Resume ID: {resume_id}")
                        found = True
                if not found:
                    print("- None found or metadata missing")

                print("\n[DEBUG] Source Documents Raw Output:")
                for i, doc in enumerate(source_docs, 1):
                    print(f"--- Document {i} ---")
                    print(f"Content Preview: {doc.page_content[:200]}...")
                    print(f"Metadata: {doc.metadata}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")

    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")


chat_history = []

def chatbot_interface(query, history=[]):
    global chat_history
    result = qa.invoke({"question": query, "chat_history": chat_history})
    
    answer = result['answer']
    source_docs = result.get('source_documents', [])

    resume_ids = []
    for doc in source_docs:
        rid = doc.metadata.get("resume_id", None)
        if rid:
            resume_ids.append(str(rid))

    chat_history.append((query, answer))

    return answer, gr.update(choices=resume_ids)

def show_full_resume(resume_id):
    return get_resume_by_id(resume_id)

def launch_gradio_chat():
    global qa  # ✅ So it's accessible in chatbot_interface
    qa = create_qa_chain()  # ✅ Initialize before launching Gradio
    with gr.Blocks() as demo:
        gr.Markdown("## 💼 Resume Search Chatbot")
        chatbot = gr.Textbox(label="Ask your question", placeholder="e.g. Python developer with Azure experience")
        output = gr.Textbox(label="Bot Response", lines=6)
        
        resume_selector = gr.Dropdown(label="Select a Resume ID to View Full Resume", choices=[])
        full_resume = gr.Textbox(label="Full Resume Content", lines=15, interactive=False)

        chatbot.submit(fn=chatbot_interface, inputs=chatbot, outputs=[output, resume_selector])
        resume_selector.change(fn=show_full_resume, inputs=resume_selector, outputs=full_resume)

    demo.launch()

    
if __name__ == "__main__":
    try:
        print("🤖 Welcome to Resume Search Chatbot")
        print("Choose mode to continue:")
        print("1. Terminal (CLI)")
        print("2. Web Interface (Gradio)\n")

        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            cli_chatbot()
        elif choice == "2":
            launch_gradio_chat()
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    except KeyboardInterrupt:
        print("\nGoodbye!")
