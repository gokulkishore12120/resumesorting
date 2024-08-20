import hashlib
import requests
import pandas as pd
from io import StringIO
import os
from datetime import datetime

active_sessions = {}
def fetch_credentials(sheet_url):
    response = requests.get(sheet_url)
    data = response.text
    # print(data)
    df = pd.read_csv(StringIO(data))
    # print(df)
    # for _,row in df.iterrows():
    #     print(row['username'],row['password'])
    return {row['username']: row['password'] for _, row in df.iterrows()}

#https://docs.google.com/spreadsheets/d/10yxVHVyZusKZO3XR0P-35fPUulNOXWVnG1FadIyy1uU/export?format=csv
GOOGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1p5E4ZkjxtOMNc764rjIWWcKdMgAQszhIGMDz92O9chY/pub?gid=0&single=true&output=csv'
user_credentials = fetch_credentials(GOOGLE_SHEET_URL)
#print(user_credentials)
def authenticate(username, password):
    hashed_password = str(password)
    if username in user_credentials and str(user_credentials[username]) == hashed_password:
        print("coming")
        log_activity(username, 'loggedin')
        return True
    print("not coming")
    return False

def admin_auth(username,email,password):
    hashed_password = str(password)
    if email in user_credentials and str(user_credentials[email]==hashed_password):
        print("admin coming")
        log_activity(username,'loggedin')
        return True
    return False

def create_session(username):
    session_id = hashlib.sha256(os.urandom(24)).hexdigest()
    active_sessions[session_id] = username
    return session_id

def validate_session(session_id):
    return session_id in active_sessions

def logout(session_id):
    username = active_sessions.get(session_id)
    if session_id in active_sessions:
        log_activity(username,'loggedout')
        del active_sessions[session_id]

def log_activity(username, log):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('updated_logs.csv', mode='a') as file:
        file.write(f"{username},{log},{now}\n")
    print(f"{username} {log} at {now}")


    
