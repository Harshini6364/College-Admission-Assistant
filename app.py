import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ---------------------------
# LOAD ENV VARIABLES
# ---------------------------
load_dotenv()

# ---------------------------
# STREAMLIT UI SETTINGS
# ---------------------------
st.set_page_config(page_title="College Admission Assistant")

st.title("🎓 College Admission Assistant")
st.write("Ask questions about admissions, fees, hostel, scholarships, etc.")
st.subheader("Example questions:")
st.write("1. What is the admission process?")
st.write("2. What documents are required?")

# ---------------------------
# LOAD PDF FILES
# ---------------------------
documents = []
pdf_folder = "data"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

# ---------------------------
# SPLIT TEXT INTO CHUNKS
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# ---------------------------
# EMBEDDING MODEL
# ---------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# PINECONE SETUP
# ---------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "college-chatbot"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # MiniLM embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# ---------------------------
# VECTOR STORE (PINECONE)
# ---------------------------
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=index_name
)

retriever = vectorstore.as_retriever()

# ---------------------------
# GROQ LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ---------------------------
# SESSION STATE (CHAT + CACHE)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "cache" not in st.session_state:
    st.session_state.cache = {}

# ---------------------------
# USER INPUT
# ---------------------------
query = st.chat_input("Ask your question")

if query:

    # Start timer
    start = time.time()

    # store user message
    st.session_state.messages.append(("user", query))

    # CACHE CHECK
    if query in st.session_state.cache:

        #st.info("⚡ Response from CACHE")
        response_text = st.session_state.cache[query]

    else:

        #st.info("🧠 Fresh response from Pinecone + LLM")

        # retrieve relevant docs
        relevant_docs = retriever.invoke(query)

        # combine context
        context = "\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # prompt
        prompt = f"""
        You are a helpful college admission assistant.

        Answer only using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # LLM response
        response = llm.invoke(prompt)

        response_text = response.content

        # store in cache
        st.session_state.cache[query] = response_text
    # End timer
    end = time.time()

    # Calculate response time
    response_time = round(end - start, 2)

    # Final response with timing
    final_response = f"""
{response_text}

---
⏱️ **Response Time:** `{response_time} sec`
"""

    # Store assistant response
    st.session_state.messages.append(
        ("assistant", final_response)
    )

# ---------------------------
# DISPLAY CHAT UI
# ---------------------------
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)