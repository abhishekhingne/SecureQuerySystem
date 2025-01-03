import streamlit as st
import sqlite3

# Initialize session state for login and chat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def validate_user(conn, username, password):
    cursor = conn.cursor()
    p = cursor.execute('''select * from users where email="{}"'''.format(username))
    data = p.fetchone()
    try:
        return True if data[1] == password else False
    except:
        return False

# Sidebar for login
with st.sidebar:
    st.header("Login")
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            with sqlite3.connect('/home/intellect/tutorial/SecureQuerySystem/backend/src/data/users.db') as conn:
                if validate_user(conn, username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
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

# Main chat window
if st.session_state.logged_in:
    st.title("💬 Assistant")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        url = 'https://www.w3schools.com/python/demopage.php'

        #response = requests.post(api_url + "/simple_rag/", json=payload).json()

        st.chat_message("assistant").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": prompt})


else:
    st.title("Please log in to access the chatbot.")

