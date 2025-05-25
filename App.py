import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Dict, Any
import requests
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Custom wrapper for Groq to make it LangChain compatible
class GroqWrapper(LLM):
    client: Any
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=self.temperature,
            **kwargs
        )
        return response.choices[0].message.content

# Initialize global variables
vectorstore = None
qa_chain = None
groq_llm = None

def upload_pdf(file):
    global vectorstore, qa_chain, groq_llm
    
    try:
        # Initialize Groq LLM wrapper
        groq_llm = GroqWrapper(client=Groq(api_key=GROQ_API_KEY))
        
        # PDF Text Extraction
        text = "".join(
            page.extract_text() or ""
            for page in PdfReader(file).pages
        )
        if not text.strip():
            return "Error: No readable text found in PDF"

        # Text Chunking
        texts = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_text(text)

        # Using HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        vectorstore = FAISS.from_texts(texts, embeddings)

        # QA System Initialization
        qa_chain = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        return "PDF processed successfully!"
    except Exception as e:
        return f"Error: {str(e)}"



def ask_question(query):
    if qa_chain is None:
        return "Please upload a PDF first.", ""

    result = qa_chain(query, return_only_outputs=False)
    answer = result["result"]
    sources = result["source_documents"]

    source_text = "\n\n".join([f"---\n{doc.page_content[:500]}..." for doc in sources])
    return answer, source_text or "No sources found."

def summarize_pdf(num_points: int = 6) -> str:
    """
    Summarizes the uploaded PDF using the Groq LLM with a creative prompt.
    
    Args:
        num_points (int): Number of bullet points for the summary (default: 6).
    
    Returns:
        str: The summary or an error message.
    """
    global vectorstore, groq_llm

    if vectorstore is None:
        return "Please upload a PDF first."

    try:
        docs = vectorstore.similarity_search("summary", k=5)
        if not docs:
            return "No content found to summarize."

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "Imagine you are a passionate science communicator tasked with revealing the essence of a groundbreaking research paper.\n"
            f"Craft a captivating summary in {num_points} vivid bullet points that not only highlights the core discoveries but also paints a clear picture of their significance.\n"
            "Make it engaging, insightful, and accessible to a curious reader eager to grasp the impact of this work.\n\n"
            f"Here is the paper content:\n{context}\n\n"
            "Your inspired summary:"
        )

        if groq_llm is None:
            from groq import Groq
            groq_llm = GroqWrapper(
                client=Groq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.3-70b-versatile"
                ),
                model_name="llama-3.3-70b-versatile"
            )

        summary = groq_llm(prompt)
        return summary.strip()
    
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def find_similar_papers():
    if vectorstore is None:
        return "Please upload a PDF first."

    docs = vectorstore.similarity_search("summary", k=1)
    query_text = docs[0].page_content[:1000]  # Shorten if needed

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_text}&limit=3&fields=title,abstract,url"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return "Error fetching from Semantic Scholar."

    data = response.json()
    results = data.get("data", [])

    return "\n\n".join([f"**{r['title']}**\n{r['abstract']}\nLink: {r['url']}" for r in results])

# Gradio UI
css = '''
body, .gradio-container {
  margin: 0; padding: 0; min-height: 100vh;
  background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
  color: #eee;
}
.header {
  width: 100%; padding: 20px; text-align: center;
  font-size: 2.8rem; font-weight: bold;
  background: linear-gradient(90deg,#ff8a00,#e52e71);
  -webkit-background-clip: text; color: transparent;
}
.content {
  display: flex; flex-wrap: wrap; padding: 20px;
  gap: 20px; justify-content: center;
}
.sidebar, .main {
  background: rgba(255,255,255,0.05);
  border-radius: 12px; padding: 20px;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.8s ease-in-out;
  max-height: 70vh; overflow-y: auto;
  flex: 1 1 300px;
}
.footer {
  display: flex; flex-wrap: wrap; gap: 20px;
  padding: 15px; justify-content: center;
}
button {
  transition: transform .15s ease-in-out, box-shadow .15s;
  padding: 10px 20px;
}
button:hover {
  transform: scale(1.05);
  box-shadow: 0 0 8px rgba(255,255,255,0.4);
}
@keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
'''

with gr.Blocks(css=css) as demo:
    gr.Markdown("<div class='header'>🧠 AI Research Assistant</div>")

    with gr.Row(elem_classes=["content"]):
        with gr.Column(elem_classes=["sidebar"]):
            gr.Markdown("### Upload & Process PDF")
            file_upload = gr.File(file_types=['.pdf'], label="Upload PDF")
            upload_btn  = gr.Button("📂 Process PDF", variant="primary")
            status      = gr.Textbox(label="Status", interactive=False)

        with gr.Column(elem_classes=["main"]):
            gr.Markdown("### Ask a Question")
            question    = gr.Textbox(placeholder="Type your question…", label="Question")
            ask_btn     = gr.Button("❓ Submit", variant="secondary")
            answer      = gr.Textbox(label="Answer", lines=6, interactive=False)
            citations   = gr.Textbox(label="Sources", lines=4, interactive=False)

    with gr.Row(elem_classes=["footer"]):
        with gr.Column():
            summary_btn    = gr.Button("✍️ Summarize", variant="primary")
            summary_output = gr.Textbox(label="Summary", lines=4, interactive=False)
        with gr.Column():
            similar_btn    = gr.Button("🔍 Similar Papers", variant="primary")
            similar_output = gr.Textbox(label="Similar Papers", lines=4, interactive=False)

    upload_btn.click(upload_pdf, inputs=file_upload, outputs=status)
    ask_btn.click(ask_question, inputs=question, outputs=[answer, citations])
    summary_btn.click(summarize_pdf, outputs=summary_output)
    similar_btn.click(find_similar_papers, outputs=similar_output)

if __name__ == "__main__":
    demo.launch()
