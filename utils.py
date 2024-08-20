import re
import numpy as np
import spacy
import docx2txt
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
import torch
import pdfplumber
import mimetypes

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

nlp = spacy.load("en_core_web_sm")


def read_pdf_text(uploaded_file):
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type != 'application/pdf':
        raise ValueError("Uploaded file is not a PDF")
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""


def read_docx_text(uploaded_file):
    return docx2txt.process(uploaded_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def compute_cosine_similarity(vector1, vector2):
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def custom_matching_percentage(job_description, resume_texts):
    job_desc_embedding = get_bert_embedding(preprocess_text(job_description))
    
    resume_embeddings = []
    for resume_text in resume_texts:
        resume_embeddings.append(get_bert_embedding(preprocess_text(resume_text)))
    
    similarities = []
    for resume_embedding in resume_embeddings:
        similarity = compute_cosine_similarity(job_desc_embedding, resume_embedding)
        similarities.append(similarity)
    
    similarities = np.array(similarities) * 100
    return similarities

def extract_entities_keywords(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return entities, keywords

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