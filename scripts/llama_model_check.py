import ollama
import os

# Set the Ollama path to your models directory
os.environ["OLLAMA_PATH"] = r"C:\Users\Hp\.ollama"

try:
    # Retrieve available models from Ollama
    models_response = ollama.list()
    print("Available Models:", models_response)  # Debugging to view the structure

    # Check if the specific model exists
    model_names = [m.model for m in models_response['models']]  # Access `model` attribute using dot notation
    if 'llama3:8b-instruct-q4_0' not in model_names:
        raise ValueError("Ollama model 'llama3:8b-instruct-q4_0' not found")
    else:
        print("Model 'llama3:8b-instruct-q4_0' found and ready to use!")

except Exception as e:
    print(f"An error occurred: {e}")