import streamlit as st
import pandas as pd
import authentication
import utils
import zipfile
import os
from PIL import Image
import pytesseract
import fitz

# Create directories if they don't exist
os.makedirs("job_descriptions", exist_ok=True)
os.makedirs("resumes_uploaded", exist_ok=True)

def login():
    st.title("Login")
    with st.form("Form1"):
        username = st.text_input("Username", placeholder="Enter Your Sacha Username")
        password = st.text_input("Password", type="password", placeholder="Enter Your Password")

        if st.form_submit_button("Login"):
            if authentication.authenticate(username, password):
                session_id = authentication.create_session(username)
                st.session_state['session_id'] = session_id
                st.session_state['username'] = username
                st.session_state['page'] = "home"
            else:
                st.error("Invalid username or password.")

def logout():
    session_id = st.session_state.get("session_id", None)
    if session_id:
        authentication.logout(session_id)
    st.session_state['page'] = "login"
    st.session_state.clear()
    st.success("You are successfully logged out.")

def show_logs():
    session_id = st.session_state.get("session_id", None)
    if session_id is None:
        st.error("Unauthorized access. Please login.")
        st.session_state['page'] = 'login'
        st.stop()
    st.subheader("Activity Logs")
    st.sidebar.markdown(f"""
    <div style='
        border: 2px solid yellow; 
        padding: 10px; 
        border-radius: 5px;
        user-select:none; 
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
        color: black;
        background-color: #f0f0f0;
    '>
        👤 Hello, {st.session_state['username']}
    </div>
""", unsafe_allow_html=True)
    logs_df = pd.read_csv("updated_logs.csv")
    st.write(logs_df)

def home():
    session_id = st.session_state.get("session_id", None)
    if session_id is None:
        st.error("Unauthorized access. Please login.")
        st.session_state['page'] = 'login'
        st.stop()

    if session_id not in authentication.active_sessions:
        st.error("Session expired or invalid. Please login again.")
        st.session_state['page'] = 'login'
        st.stop()

    st.sidebar.markdown(f"""
        <div style='
            border: 2px solid yellow; 
            padding: 10px; 
            border-radius: 5px;
            user-select:none; 
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
            color: black;
            background-color: #f0f0f0;
        '>
            👤 Hello, {st.session_state['username']}
        </div>
    """, unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        option_job = st.radio("Select Job Description:", ('Upload File', 'Select Previous', 'Image'))
    
    with col4:
        option = st.radio("Upload resumes as:", ('Individual Files', 'Zip File'))

    col1, col2 = st.columns(2)
    job_description_file = None
    job_description_key = None
    job_description = None
    job_description_filename = None
    job_descriptions_df = pd.read_csv("job_descriptions.csv") if os.path.isfile("job_descriptions.csv") else pd.DataFrame(columns=["Key", "Filename"])
    keys = job_descriptions_df['Key'].unique()

    with col1:
        if option_job == 'Upload File':
            job_description_file = st.file_uploader("Upload Job Description", type=['pdf', 'txt', 'docx'])
            job_description_key = st.text_input("Enter a key for the job description")
        elif option_job == 'Image':
            job_description_file = st.file_uploader("Upload Job Description Image", type=['jpg', 'png'])
        else:
            job_key = st.selectbox("Select a key:", keys)
            if job_key:
                job_description_key = job_key
                job_description_filename = job_descriptions_df.loc[job_descriptions_df['Key'] == job_key, 'Filename'].iloc[0]
                
                if os.path.isfile(job_description_filename):
                    file_extension = job_description_filename.split('.')[-1].lower()
                    
                    if file_extension == 'txt':
                        with open(job_description_filename, "r", encoding="utf-8", errors="ignore") as f:
                            job_description = f.read()
                    
                    elif file_extension == 'pdf':
                        with open(job_description_filename, "rb") as f:
                            job_description = utils.read_pdf_text(f)
                    
                    elif file_extension == 'docx':
                        job_description = utils.read_docx_text(job_description_filename)
                    
                    elif file_extension in ['png', 'jpeg', 'jpg']:
                        image = Image.open(job_description_filename)
                        job_description = pytesseract.image_to_string(image)
                    
                    else:
                        st.error("Unsupported file format.")
                else:
                    st.error(f"File {job_description_filename} not found.")
            else:
                st.warning("Please select a job description key.")

    with col2:
        if option == 'Individual Files':
            uploaded_files = st.file_uploader("Upload Resumes", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
        else:
            uploaded_zip = st.file_uploader("Upload Resumes Zip", type=['zip'])

    if job_description_file and job_description_key:
        job_desc_filename = f"job_descriptions/{job_description_key}_{job_description_file.name}"
        with open(job_desc_filename, "wb") as f:
            f.write(job_description_file.getvalue())

        if job_description_file.type == "text/plain":
            job_description = job_description_file.getvalue().decode("utf-8")
        elif job_description_file.type == "application/pdf":
            job_description = utils.read_pdf_text(job_description_file)
        elif job_description_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            job_description = utils.read_docx_text(job_description_file)

        new_entry = pd.DataFrame([[job_description_key, job_desc_filename]], columns=["Key", "Filename"])
        job_descriptions_df = pd.concat([job_descriptions_df, new_entry], ignore_index=True)
        job_descriptions_df.to_csv("job_descriptions.csv", index=False)

    elif option_job == 'Image' and job_description_file:
        image = Image.open(job_description_file)
        job_description = pytesseract.image_to_string(image)
        job_description_key = st.text_input("Enter a key for the job description from image")

        if job_description_key:
            job_desc_filename = f"job_descriptions/{job_description_key}_{job_description_file.name}"
            with open(job_desc_filename, "wb") as f:
                f.write(job_description_file.getvalue())

            new_entry = pd.DataFrame([[job_description_key, job_desc_filename]], columns=["Key", "Filename"])
            job_descriptions_df = pd.concat([job_descriptions_df, new_entry], ignore_index=True)
            job_descriptions_df.to_csv("job_descriptions.csv", index=False)

    resumes = {}
    if option == 'Individual Files' and uploaded_files:
        for uploaded_file in uploaded_files:
            resume_filename = f"resumes_uploaded/{uploaded_file.name}"
            with open(resume_filename, "wb") as f:
                f.write(uploaded_file.getvalue())

            resume_text = ""
            if uploaded_file.type == "text/plain":
                resume_text = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                resume_text = utils.read_pdf_text(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = utils.read_docx_text(uploaded_file)
            
            resumes[resume_filename] = resume_text
    elif option == 'Zip File' and uploaded_zip:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                st.spinner('Extracting files...')
                zip_ref.extractall("resumes_uploaded")

            for root, dirs, files in os.walk("resumes_uploaded"):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    resume_text = ""
                    if file_name.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as file:
                            resume_text = file.read()
                    elif file_name.endswith(".pdf"):
                        with open(file_path, "rb") as file:
                            resume_text = utils.read_pdf_text(file)
                    elif file_name.endswith(".docx"):
                        resume_text = utils.read_docx_text(file_path)
                    
                    resumes[file_path] = resume_text

    if job_description and resumes:
        threshold_percentage = st.number_input("Minimum Matching Percentage to Consider", min_value=0, max_value=100, value=50)
        if st.button("Match", key="match_button"):
            with st.spinner('Matching Resumes...'):
                nlp_percentages = utils.custom_matching_percentage(job_description, list(resumes.values()))
                feedback = utils.generate_feedback(job_description, resumes)

            results_consider = []
            results_ignore = []
            for name, nlp_percentage in zip(resumes.keys(), nlp_percentages):
                consideration = "Consider" if nlp_percentage >= threshold_percentage else "Ignore"
                if consideration == "Consider":
                    results_consider.append({
                        "Resume Name": name,
                        "Matching Percentage": nlp_percentage,
                        "Consideration": consideration,
                        "Feedback": feedback[name]
                    })
                else:
                    results_ignore.append({
                        "Resume Name": name,
                        "Matching Percentage": nlp_percentage,
                        "Consideration": consideration,
                        "Feedback": feedback[name]
                    })

            df_consider = pd.DataFrame(results_consider)
            df_ignore = pd.DataFrame(results_ignore)

            def highlight_row(row):
                color = 'color: red' if row.Consideration == 'Ignore' else 'color: yellow'
                return [color] * len(row)

            styled_df_consider = df_consider.style.apply(highlight_row, axis=1)
            styled_df_ignore = df_ignore.style.apply(highlight_row, axis=1)

            st.subheader("Consider")
            st.write(styled_df_consider)

            csv_consider = df_consider.to_csv(index=False)
            st.download_button(label="Download Consider Table", data=csv_consider, file_name="consider.csv", mime="text/csv", key="consider_button")

            st.subheader("Ignore")
            st.write(styled_df_ignore)

            csv_ignore = df_ignore.to_csv(index=False)
            st.download_button(label="Download Ignore Table", data=csv_ignore, file_name="ignore.csv", mime="text/csv", key="ignore_button")

def settings():
    st.header("Settings")
    with st.form(key='settings_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        role = st.selectbox("Role", ["Select Role", "HR", "Admin"])
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        st.success(f"Settings updated for {name}. Role: {role}")

def main():
    st.set_page_config(page_title="Analyzer", page_icon=":mag:", layout="wide")
    st.markdown("<h1 style='color: yellow; text-align: center; border: 2px solid yellow; padding: 10px; user-select:none;'>+++ SACHA </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; user-select:none'>&#128269; || RESUME ANALYZER ||</h3>", unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

    if st.session_state['page'] == "login":
        login()
    elif st.session_state['page'] == "home":
        home()
    elif st.session_state['page'] == "logs":
        show_logs()
    elif st.session_state['page'] == "settings":
        settings()
    
    selected_option = st.sidebar.selectbox("Menu", ["Select Option", "Home", "Logs", "Logout", "Settings"])
    if selected_option == "Home":
        st.session_state['page'] = 'home'
    elif selected_option == "Logs":
        st.session_state['page'] = 'logs'
    elif selected_option == "Logout":
        logout()
    elif selected_option == "Settings":
        st.session_state['page'] = 'settings'

if __name__ == "__main__":
    main()

