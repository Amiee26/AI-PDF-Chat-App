import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
from streamlit_mic_recorder import mic_recorder
import os
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()
st.set_page_config(
    page_title="AI PDF Chat App",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stTextInput > div > div > input {
    background-color: #262730;
    color: white;
}

.stButton button,
.stDownloadButton button {

    background-color: #1E1E1E !important;
    color: white !important;
    border-radius: 10px;
    border: none !important;
}

.stButton button:hover,
.stDownloadButton button:hover {

    background-color: #1E1E1E !important;
    color: white !important;
    border: none !important;
}

.stButton button:focus,
.stDownloadButton button:focus {

    background-color: #1E1E1E !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant.
    Use the provided PDF context to answer the question.
    If multiple PDFs are uploaded:
    - compare them
    - identify similarities and differences
    - summarize each separately when asked

    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output("summary.pdf")

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    reply = response["output_text"]

    st.session_state.chat_history.append(
        (user_question, reply)
    )

    for question, answer in st.session_state.chat_history:
        st.markdown("### You:")
        st.write(question)

        st.markdown("### AI:")
        st.write(answer)
    create_pdf(reply)

    with open("summary.pdf", "rb") as file:
        st.download_button(
            label="⬇ Download Response PDF",
            data=file,
            file_name="summary.pdf",
            mime="application/pdf"
        )

def main():
    st.title("📄 AI PDF Chat App")
    st.write("Upload PDFs and ask questions instantly.")

    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip() == "":
                        st.error("No readable text found in PDF.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    if len(text_chunks) == 0:
                        st.error("Text chunks are empty.")
                        return

                    get_vector_store(text_chunks)
                    st.success("PDFs Processed Successfully!")

            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()