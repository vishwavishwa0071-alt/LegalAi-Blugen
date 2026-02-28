# ⚖️ LegalAI Blugen

A powerful **Legal AI Chatbot** built with Streamlit and Google Gemini (RAG-based), capable of answering legal queries with source attribution and PDF highlighting.

## 🚀 Features

- 🤖 **AI-Powered Legal Q&A** — Uses Google Gemini via LangChain RAG pipeline
- 📄 **PDF Preview with Highlighting** — Highlights the exact paragraph referenced in the response
- 🗂️ **Source Attribution** — Displays category, filename, and folder for every retrieved chunk
- 🌙 **Dark Mode UI** — Glassmorphism-inspired, split-pane Streamlit interface
- 🔍 **Unified Vector Store** — FAISS-based vector DB over 300+ Tamil Nadu legal PDFs

## 🧱 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Vector Store | FAISS |
| PDF Parsing | PyMuPDF (fitz), PyPDF |
| Embeddings | Google Generative AI Embeddings |
| Framework | LangChain |

## 📁 Project Structure

```
RAG/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (NOT committed — see below)
├── data/                   # Category mapping & metadata (gitignored)
├── pdf/                    # Source legal PDFs (gitignored)
└── unified_vector_store/   # FAISS index (gitignored)
```

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/vishwavishwa0071-alt/LegalAi-Blugen.git
cd LegalAi-Blugen
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 4. Add your PDFs and build the vector store
Place your legal PDFs in the `pdf/` folder and run the ingestion script to build the FAISS vector store.

### 5. Run the app
```bash
streamlit run app.py
```

## 🔒 Security Note

The `.env` file containing your `GOOGLE_API_KEY` is **gitignored** and will never be pushed to GitHub. Always keep your API keys private.

## 📜 License

This project is for educational and research purposes related to Tamil Nadu legal documents.
