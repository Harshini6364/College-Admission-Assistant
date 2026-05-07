import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit page settings
st.set_page_config(page_title="College Admission Assistant")

st.title("🎓 College Admission Assistant")
st.write("Ask questions about admission, fees, hostel, scholarships, etc.")

# Load PDF documents
documents = []

pdf_folder = "data"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever()

# Load Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
query = st.chat_input("Ask a question")

if query:

    # Add user message
    st.session_state.messages.append(("user", query))

    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Combine document contents
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Prompt
    prompt = f"""
    You are a helpful college admission assistant.

    Answer the question only using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    # Generate response
    response = llm.invoke(prompt)

    # Store assistant response
    st.session_state.messages.append(
        ("assistant", response.content)
    )

# Display chat messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)