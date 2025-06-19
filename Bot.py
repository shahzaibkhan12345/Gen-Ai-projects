#Importing Necessary libraries
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("My ChatBot App")

st.write("This is a simple chatbot application built with Streamlit.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all messages in the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource
def get_vectorestore(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
        return None
    try:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = [PyPDFLoader("temp.pdf")]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        ).from_loaders(loader)
        os.remove("temp.pdf")  # Clean up temporary file
        return index.vectorstore
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Add file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Get user input
prompt = st.chat_input("Type your message here...")
system_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a very smart assistant and you always give the best answer to the user query. Answer the following question: {question}. No small talk, just be precise and accurate in your answer.")
])
model='llama3-8b-8192'

groq_chat=ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=model,
)
try:
    vectorestore = get_vectorestore(uploaded_file)
    if vectorestore is None:
        st.error("Vectorstore is not available. Please check your configuration.")
    else:
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorestore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        if prompt:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from chain
            result = chain({'query': prompt})
            response = result["result"]

            # Append assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
except Exception as e:
    st.error(f"An error occurred: {e}")