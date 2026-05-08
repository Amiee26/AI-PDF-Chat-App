# AI PDF Chat App

An AI-powered PDF chatbot built using Streamlit, LangChain, FAISS, and Groq API. Users can upload PDF files, ask questions, generate summaries, compare multiple PDFs, and download responses as PDF files.

## Features

* Chat with uploaded PDF documents
* AI-generated PDF summaries
* Multiple PDF support
* PDF comparison
* Download responses as PDF
* Chat history support
* Modern dark UI using Streamlit

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS Vector Store
* HuggingFace Embeddings
* Groq API (LLM)
* PyPDF2

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPO_LINK
cd AI_pdf_chat_App
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
AI_pdf_chat_App/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── faiss_index/
```

## Future Improvements

* Voice assistant support
* OCR for scanned PDFs
* Multi-language support
* User authentication

## Author
Ambika Devkar
