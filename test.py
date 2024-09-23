import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_fireworks import FireworksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.environ.get("FIREWORKS_API_KEY")
groq_api_key = os.getenv('GROQ_API')

# Streamlit App title
st.title("Health Consultant")

# Initialize the Language Model (LLM)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define system prompt template for the QA task
system_prompt = """
unless user say name don't retrive information from provided content. Retrivie information related to name from content.
Check User information, health history and must provide the most accurate response based on the question about symptoms, mood, and concerns.
<context>
{context}
<context>
Questions: {input}
"""

# Contextualize question system prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Set up the prompt templates
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Embedding initialization function
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
        st.session_state.loader = CSVLoader(file_path='data/userdata.csv', csv_args={'delimiter': ','})
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:250])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Function to create the retrieval chain with history-awareness
def create_qa_chain_with_history():
    # Retrieve documents and embeddings if not already loaded
    if "vectors" not in st.session_state:
        vector_embedding()

    retriever = st.session_state.vectors.as_retriever()

    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    # Create the retrieval-augmented QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# Initialize or retrieve chat history from the session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user's question
prompt1 = st.text_input("Enter Your Question :")

# Button to trigger the document embedding process
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Process user input and maintain chat history
if prompt1:
    rag_chain = create_qa_chain_with_history()
    start_time = time.process_time()
    
    # Call the retrieval chain with the current question and chat history
    response = rag_chain.invoke({"input": prompt1, "chat_history": st.session_state.chat_history})
    
    # Calculate response time
    response_time = time.process_time() - start_time
    st.write(f"Response time: {response_time:.2f} seconds")
    
    # Display the response from the LLM
    st.write(response['answer'])

    # Update chat history with the latest question and answer
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=prompt1),
            AIMessage(content=response['answer']),
        ]
    )

    # Show relevant document context for transparency
    with st.expander("Document Similarity Search"):
        if "context" in response:
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
