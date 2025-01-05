import streamlit as st
import sqlite3
import requests

# Initialize session state for login and chat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "company_name" not in st.session_state:
    st.session_state.company_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def validate_user(conn, username, password):
    cursor = conn.cursor()
    p = cursor.execute('''select * from users where email="{}"'''.format(username))
    data = p.fetchall()
    print(data)
    try:
        return (data, True) if data[0][1] == password else (data, False)
    except:
        return (data, False)

# Sidebar for login
with st.sidebar:
    st.header("Login")
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            with sqlite3.connect('/home/intellect/tutorial/SecureQuerySystem/backend/src/data/users.db') as conn:
                result = validate_user(conn, username, password)
                data = result[0]
                check = result[1]
                if check:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.company_name = [i[-1] for i in data]
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Invalid username or password.")
    else:
        st.write(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.info("You have been logged out.")

username = st.session_state.username
company_name = st.session_state.company_name

# Main chat window
if st.session_state.logged_in:
    st.title("💬 Assistant")

    # st.session_state["data"] = data
    # username = st.session_state["data"][0][0]
    # company_name = [i[-1] for i in st.session_state["data"]]

    st.chat_message("assistant").write("""You're logged in as {}. 
                                       You've access to the following compny reports: {}. 
                                       How can I help you?""".format(username, ", ".join(company_name)))
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        url = 'https://www.w3schools.com/python/demopage.php'

        payload = {"query": prompt, "company_name": st.session_state.company_name, 
                   "email": st.session_state.username}
        api_url = "http://0.0.0.0:8000"
        response = requests.post(api_url + "/chat/", json=payload).json()

        st.chat_message("assistant").write(response["generation"].replace("\n", " ").replace("### Answer: ", ""))
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["generation"]})


else:
    st.title("Please log in to access the chatbot.")

