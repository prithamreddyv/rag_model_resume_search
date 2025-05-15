# import pandas as pd
# import os

# def preprocess_data():
#     # Load datasets
#     resumes_df = pd.read_csv("C:/Users/Hp/Documents/resume_search/data/synthetic-resumes.csv")
#     # jobs_df = pd.read_csv("../data/sampled-job-titles.csv")

#     # For resumes: Include ID in the text
#     resumes_df["text"] = "ID:" + resumes_df["ID"].astype(str) + " | RESUME: " + resumes_df["Resume"].str[:500]
    
#     # For jobs: Create synthetic IDs
#     # jobs_df["Job_ID"] = "JOB_" + jobs_df.index.astype(str)
#     # jobs_df["text"] = "ID:" + jobs_df["Job_ID"] + " | JOB: " + jobs_df["Job Title"] + ". " + jobs_df["Job Description"].str[:500]

#     # Create outputs directory if it doesn't exist
#     os.makedirs("../outputs", exist_ok=True)

#     # Save with UTF-8 encoding to handle special characters
#     with open("../outputs/combined_texts.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(resumes_df["text"].tolist()))
#                 #  + jobs_df["text"].tolist()))

#     print("Preprocessing completed successfully!")

# if __name__ == "__main__":
#     preprocess_data()

import pandas as pd
import os

def preprocess_data():
    # Define the absolute path for the outputs directory
    outputs_path = "C:/Users/Hp/Documents/resume_search/outputs"
    
    # Load the dataset
    try:
        resumes_df = pd.read_csv("C:/Users/Hp/Documents/resume_search/data/synthetic-resumes.csv")
    except FileNotFoundError:
        print("Error: The file 'synthetic-resumes.csv' could not be found. Please check the file path.")
        return
    
    # Include ID in the text for resumes
    resumes_df["text"] = "ID:" + resumes_df["ID"].astype(str) + " | RESUME: " + resumes_df["Resume"].str[:500]
    
    # Create outputs directory if it doesn't exist
    try:
        os.makedirs(outputs_path, exist_ok=True)
        print(f"Directory '{outputs_path}' created successfully!")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    # Save the combined texts with UTF-8 encoding
    try:
        output_file_path = os.path.join(outputs_path, "combined_texts.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(resumes_df["text"].tolist()))
        print(f"File saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        return

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()