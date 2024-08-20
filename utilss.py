import re
import zipfile
import os
import numpy as np
import pandas as pd
import spacy
import docx2txt
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def read_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx_text(uploaded_file):
    return docx2txt.process(uploaded_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def compute_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def custom_matching_percentage(job_description, resume_texts):
    job_desc_embedding = get_bert_embeddings(job_description)
    
    resume_embeddings = []
    for resume_text in resume_texts:
        resume_embedding = get_bert_embeddings(resume_text)
        resume_embeddings.append(resume_embedding)
    
    similarities = []
    for resume_embedding in resume_embeddings:
        similarity = compute_cosine_similarity(job_desc_embedding, resume_embedding)
        similarities.append(similarity)
    
    similarities = np.array(similarities) * 100
    return similarities

def extract_job_openings_from_pdf(text):
    job_openings = []
    pages = text.split("New Job Opening")
    
    for page in pages[1:]:  # Skip the first split which might be empty
        job_details = {}
        lines = page.split('\n')
        
        for line in lines:
            if "Job ID" in line:
                job_details['Job ID'] = line.split("Job ID")[1].strip()
            elif "Job Title" in line:
                job_details['Job Title'] = line.split("Job Title")[1].strip()
            elif "Skill set" in line:
                job_details['Skill Set'] = line.split("Skill set")[1].strip()
            elif "Job Description" in line:
                job_details['Job Description'] = line.split("Job Description")[1].strip()
        
        if job_details:
            job_openings.append(job_details)
    
    return job_openings

def process_resumes_from_zip(zip_file_path):
    resumes = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.pdf'):
                with zip_ref.open(file_name) as file:
                    text = read_pdf_text(file)
                    resumes[file_name] = text
            elif file_name.endswith('.docx'):
                with zip_ref.open(file_name) as file:
                    text = read_docx_text(file)
                    resumes[file_name] = text
    return resumes

def generate_feedback(job_description, resumes):
    feedback = {}
    
    job_entities, job_keywords = extract_entities_keywords(job_description)
    
    for resume_name, resume_text in resumes.items():
        resume_entities, resume_keywords = extract_entities_keywords(resume_text)
        
        missing_entities = [entity for entity in job_entities if entity not in resume_entities]
        missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_keywords]
        
        feedback[resume_name] = {
            "missing_entities": missing_entities,
            "missing_keywords": missing_keywords
        }
    
    return feedback

def save_results_to_csv(results, filename='final_result.csv'):
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)

def main():
    # Paths
    job_descriptions_path = 'path_to_your_job_descriptions.pdf'
    resumes_zip_path = 'path_to_your_resumes.zip'
    
    # Read job descriptions
    job_description_text = read_pdf_text(job_descriptions_path)
    job_openings = extract_job_openings_from_pdf(job_description_text)
    
    # Process resumes
    resumes = process_resumes_from_zip(resumes_zip_path)
    
    results = []
    
    # Match job descriptions with resumes
    for job in job_openings:
        job_id = job.get('Job ID', '')
        job_title = job.get('Job Title', '')
        job_description = job.get('Job Description', '')
        skill_set = job.get('Skill Set', '')
        
        resume_texts = [resume for resume in resumes.values()]
        matching_percentages = custom_matching_percentage(job_description, resume_texts)
        
        for idx, resume_name in enumerate(resumes.keys()):
            results.append({
                'Job ID': job_id,
                'Job Title': job_title,
                'Matched Resume': resume_name,
                'Matching Percentage': matching_percentages[idx]
            })
    
    # Save results to CSV
    save_results_to_csv(results)

if __name__ == "__main__":
    main()