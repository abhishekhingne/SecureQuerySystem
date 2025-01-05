from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from typing import List
from pydantic import BaseModel
import sqlite3
import pandas as pd
import uuid
from embeddings import Embeddings
from advanced_rag import AdvancedRAG
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Sqlite Connection
conn = sqlite3.connect('data/users.db')
cursor = conn.cursor()

# URL for Ollama Embeddings
LLM_URL = os.getenv("LLM_URL")

# Define request model
class UserRequest(BaseModel):
    email: str
    password: str
    company_name: str

class Chat(BaseModel):
    email :  str
    query : str
    company_name: List[str]

# FastAPI endpoint to add users
@app.post("/add_user/")
def add_user(request: UserRequest):
    # Perform any necessary processing or validation here
    if not request.email or not request.password or not request.company_name:
        raise HTTPException(status_code=400, detail="All fields are required")

    with sqlite3.connect('data/users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO USERS VALUES ({}, {}, {})'''.format(request.email, request.password, request.company_name))
        cursor.commit()

    # Return status
    return {"message": "User added successfully"}

# FastAPI endpoint to get users
@app.get("/get_user/{email}")
def get_user(email: str):
    with sqlite3.connect('data/users.db') as conn:
        cursor = conn.cursor()
        p = cursor.execute('''select * from users where email={}'''.format(email))
        col_names = [i[0] for i in cursor.description]
        data = p.fetchall()
    df = pd.DataFrame(data)
    df.columns = col_names
    df = df.drop("Password", axis=1)
    data = df.to_dict(orient="records")
    if len(data) == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"result": data}

@app.post("/add_document/")
async def upload_files(company_name: str = Form(...), file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    with open(f"./data/"+job_id+".pdf", "wb") as f:
        f.write(await file.read())
    file_path = f"./data/"+job_id+".pdf"
    embd = Embeddings(llm_url=LLM_URL)
    vector_store = embd.load_documents(file_path=file_path, 
                                       chunk_size=2048, chunk_overlap=200, 
                                        company_name=company_name, use_existing=True)
    
    return {"success": True, "job_id": job_id, "company_name": company_name}

# FastAPI endpoint to chat
@app.post("/chat/")
def chat(request: Chat):
    question = request.query
    company_name = request.company_name
    email = request.email
    embd = Embeddings(llm_url=LLM_URL)
    vector_store = embd.get_vector_store()
    rag = AdvancedRAG(llm_url=LLM_URL, vector_store=vector_store)
    result = rag.execute_graph(question=question, company_name=company_name, email=email)
    print(result)
    return result