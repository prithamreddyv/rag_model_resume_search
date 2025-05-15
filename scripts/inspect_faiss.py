from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def inspect_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    faiss_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../outputs2/resumes_vector_db"
    )

    # Load FAISS vector DB
    db = FAISS.load_local(
        folder_path=faiss_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve a few documents directly
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents("python developer with azure experience")

    print("\nSample Results with Metadata:")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    inspect_faiss_index()
