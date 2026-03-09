import streamlit as st
import requests

API_URL = "http://localhost:8001"

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("Agentic Personal Assistant")

# ─── Sidebar ───────────────────────────────
with st.sidebar:
    st.header("Settings")
    project_id = st.text_input("Project ID", value="project_001")
    st.divider()

    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file and st.button("Upload & Index", type="primary"):
        # 1. Upload
        with st.spinner("Uploading..."):
            res = requests.post(
                f"{API_URL}/upload/{project_id}",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            )
        
        if res.status_code == 200:
            file_id = res.json().get("file_id")
            st.success(f"Uploaded: {file_id}")

            # 2. Index
            with st.spinner("Indexing..."):
                res = requests.post(
                    f"{API_URL}/index/{project_id}",
                    json={"file_id": file_id, "chunk_size": 500, "overlap_size": 50}
                )
            
            if res.status_code == 200:
                st.success("Indexed successfully!")
                st.session_state["indexed"] = True
            else:
                st.error(f"Indexing failed: {res.json()}")
        else:
            st.error(f"Upload failed: {res.json()}")

# ─── Chat ──────────────────────────────────
st.header("Chat with your documents")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if question := st.chat_input("Ask about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

   
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/chat/{project_id}",
                json={"question": question}
            )
        
        if res.status_code == 200:
            answer = res.json().get("answer", "No answer")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {res.json()}")