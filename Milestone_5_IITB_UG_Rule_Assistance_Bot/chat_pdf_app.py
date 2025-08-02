import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Persistent state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False


# --- Dark Styling ---
def inject_css():
    st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #f8f8f2;
        }
        .title-style {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #f8f8f2;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            margin-top: 1rem;
        }
        .user-message {
            align-self: flex-start;
            background-color: #2d2f36;
            padding: 0.9rem;
            border-radius: 12px 12px 12px 0px;
            color: #f8f8f2;
            max-width: 80%;
        }
        .bot-message {
            align-self: flex-end;
            background-color: #1f1f1f;
            padding: 0.9rem;
            border-radius: 12px 12px 0px 12px;
            color: #f8f8f2;
            max-width: 80%;
        }
        .stTextInput input {
            background-color: #2d2f36;
            color: #f8f8f2;
        }
        .stButton > button {
            background-color: #6c63ff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)


# --- Core Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vector_ready = True


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the context, 
    just say "answer is not available in the context". Don't make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def handle_question(user_question):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]


# --- Main App ---
def main():
    st.set_page_config("Gemini Chat PDF", layout="centered")
    inject_css()

    st.markdown('<div class="title-style">🧠 Gemini PDF Chatbot</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your uploaded PDFs.")

    # Text input
    user_question = st.text_input("💬 Ask a question:")

    if user_question and st.session_state.vector_ready:
        response = handle_question(user_question)
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="user-message">🧑‍💻 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload + Process PDFs
    st.sidebar.title("📂 Upload PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload one or more PDF files", accept_multiple_files=True)

    if st.sidebar.button("📥 Process"):
        if pdf_docs:
            with st.spinner("🔄 Processing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.sidebar.success("✅ PDFs processed!")
        else:
            st.sidebar.warning("⚠️ Please upload at least one PDF.")


if __name__ == "__main__":
    main()
