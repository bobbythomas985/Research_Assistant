# 🔬 AI Research Companion (Groq + LangChain + FAISS)

An advanced **AI-powered research assistant** that helps you analyze academic papers, ask natural language questions, generate engaging summaries, and discover related research papers — all from a modern, tabbed **Gradio interface**.

---

## 🚀 Features

✅ **PDF Upload & Text Extraction** – Extracts full text from research papers  
✅ **Chunking & Vector Embedding** – Uses LangChain + HuggingFace embeddings for semantic search  
✅ **Groq LLM Q&A** – Powered by `llama-3.3-70b-versatile` for accurate, context-aware answers  
✅ **Cited Source References** – Displays the exact chunks used for each answer  
✅ **Research Paper Summarization** – Creates engaging, layperson-friendly summaries  
✅ **Similar Paper Discovery** – Queries arXiv API to find related academic works  
✅ **Beautiful Multi-Tab UI** – Fully custom styled with Gradio + CSS  

---

## 🛠 Tech Stack

- **Python 3.9+**
- [Gradio](https://gradio.app/) – Interactive UI framework
- [LangChain](https://www.langchain.com/) – Document processing & QA chain
- [FAISS](https://github.com/facebookresearch/faiss) – Efficient similarity search
- [HuggingFace Sentence Transformers](https://www.sbert.net/) – Embeddings (`all-mpnet-base-v2`)
- [Groq API](https://groq.com/) – High-performance LLM inference
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing
- [Feedparser](https://pypi.org/project/feedparser/) – arXiv paper search
- **Custom CSS** – Modern tabbed layout, shadows, gradients, and animations

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bobbythomas985/Research_Assistant
cd Research_Assistant
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Set Up Your API Key
Export your Groq API key as an environment variable:
**Linux / macOS**
```bash
export GROQ_API_KEY="your_api_key_here"
```
**Windows**
```powershell
setx GROQ_API_KEY "your_api_key_here"
```
Alternatively, replace the placeholder in **app.py**:
```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key")
```
### 4️⃣ Run the App
```bash
python app.py
```

---

## 🖥️ How It Works

1️⃣ **Upload a PDF**  
📄 The system extracts all text from the research paper.

2️⃣ **Process & Embed**  
🔍 Splits the extracted text into overlapping chunks and creates a **FAISS vector index** using **HuggingFace embeddings** for efficient semantic search.

3️⃣ **Ask Questions**  
❓ User questions are converted into embeddings and matched with the most relevant chunks from the document.

4️⃣ **LLM Answer Generation**  
🤖 Groq’s `llama-3.3-70b-versatile` model is used to generate accurate, context-aware answers with a custom prompt.

5️⃣ **Summarize & Discover Papers**  
📝 Generates engaging, structured summaries of the document and retrieves similar papers from **arXiv** for further reading.


## 🔮 Future Improvements

- 📚 **Multi-document support** – Build a single knowledge base from multiple PDFs  
- 🧠 **LLM Reranking** – Use cross-encoder reranking for better context selection  
- 📑 **Clickable Source References** – Jump directly to relevant sections inside the PDF  
- 🚀 **Deploy on Hugging Face Spaces / Streamlit Cloud** – Make it public and shareable  
- 🌍 **Multilingual Q&A** – Integrate translation for global research accessibility

---

> *Empowering researchers to go from papers → insights → new discoveries.*
