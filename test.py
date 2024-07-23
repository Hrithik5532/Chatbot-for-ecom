import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_fireworks import FireworksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.environ.get("FIREWORKS_API_KEY")
groq_api_key = os.getenv('GROQ_API')

st.title("Chat bot custom PDF")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding(pdf_file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
        
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # Split pages into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(chunks, st.session_state.embeddings)
        
        # Remove temporary file
        os.remove("temp.pdf")
    
    st.write("Vector Store DB Is Ready")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Process PDF"):
        vector_embedding(uploaded_file)

# Query input
prompt1 = st.text_input("Enter Your Question About the PDF")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")