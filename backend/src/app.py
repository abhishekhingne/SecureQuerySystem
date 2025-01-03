from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Initialize Sqlite Connection
conn = sqlite3.connect('data/users.db')
cursor = conn.cursor()


# Define request model
class UserRequest(BaseModel):
    email: str
    password: str
    company_name: str

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