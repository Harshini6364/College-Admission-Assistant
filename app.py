import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq

# Load API key
load_dotenv()

# Streamlit page
st.set_page_config(page_title="College Admission Assistant")

st.title("🎓 College Admission Assistant")
st.write("Ask questions about admission, fees, hostel, scholarships, etc.")

# Load PDFs
documents = []

pdf_folder = "data"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()

# Load Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append(("user", query))

    response = qa_chain.run(query)

    st.session_state.messages.append(("assistant", response))

# Display messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)