import pandas as pd
import os

def preprocess_data():
    outputs_path = "C:/Users/Hp/Documents/resume_search/outputs2"
    
    try:
        resumes_df = pd.read_csv("C:/Users/Hp/Documents/resume_search/data/synthetic-resumes.csv")
    except FileNotFoundError:
        print("Error: The file 'synthetic-resumes.csv' could not be found.")
        return

    # Remove line breaks inside resumes to keep each on a single line
    resumes_df["Resume"] = resumes_df["Resume"].astype(str).apply(lambda x: x.replace("\n", " ").replace("\r", " "))

    # Add ID and prefix
    resumes_df["text"] = "ID:" + resumes_df["ID"].astype(str) + " | RESUME: " + resumes_df["Resume"].str[:500]

    try:
        os.makedirs(outputs_path, exist_ok=True)
        print(f"Directory '{outputs_path}' created.")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    try:
        output_file_path = os.path.join(outputs_path, "combined_texts.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(resumes_df["text"].tolist()))
        print(f"File saved at: {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        return

    print("Preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
