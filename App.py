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
import urllib.parse
import feedparser  # Added for the new function

# Load environment variables
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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
    global qa_chain
    
    if qa_chain is None:
        return "Please upload a PDF first.", ""

    try:
        # Create a custom prompt template for better answers
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a detailed, accurate response with proper formatting.
        
        Context:
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        custom_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Configure the QA chain with our custom prompt
        qa_chain.combine_documents_chain.llm_chain.prompt = custom_prompt
        
        # Execute the query
        result = qa_chain({"query": query}, return_only_outputs=False)
        
        # Extract answer and sources
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Format the sources for display
        if sources:
            source_text = "\n\n---\n".join([
                f"Source {i+1}:\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}"
                for i, doc in enumerate(sources)
            ])
        else:
            source_text = "No sources cited"
            
        return answer, source_text
    
    except Exception as e:
        return f"Error processing your question: {str(e)}", ""

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

# *** Modified find_similar_papers function ONLY ***
def find_similar_papers():
    if vectorstore is None:
        return "Please upload a PDF first."

    try:
        docs = vectorstore.similarity_search("abstract or introduction", k=3)

        # Combine chunks and take the first 40 words total
        combined_text = " ".join([doc.page_content for doc in docs])
        query_text = " ".join(combined_text.split()[:40])

        # Fallback if query_text is too short or citation-heavy
        if len(query_text) < 30 or "arXiv" in query_text or "[" in query_text:
            query_text = "transformer models for abstractive text summarization"

        encoded_query = urllib.parse.quote(query_text)

        # Build arXiv API query
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=2"

        print("Querying arXiv with:", query_text)
        print("URL:", url)

        feed = feedparser.parse(url)
        entries = feed.entries

        if not entries:
            return f"No similar papers found for query: **{query_text}**"

        results = []
        for entry in entries:
            title = entry.title
            summary = entry.summary.replace('\n', ' ').strip()
            link = entry.link
            results.append(f"**{title}**\n{summary}\n🔗 {link}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Error fetching similar papers: {str(e)}"







css = '''
:root {
    --primary: #6e48aa;
    --secondary: #9d50bb;
    --accent: #4776e6;
    --dark: #1a1a2e;
    --darker: #16213e;
    --light: #f8f9fa;
    --success: #4caf50;
    --warning: #ff9800;
    --danger: #f44336;
}

body, .gradio-container {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    background: linear-gradient(135deg, var(--dark), var(--darker));
    color: var(--light);
    min-height: 100vh;
}

.header {
    text-align: center;
    padding: 1.5rem 0;
    margin-bottom: 2rem;
    color: white;                      /* Make text white */
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 1px;
    font-style: italic;               /* Make it italic */
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}


.nav-tabs {
    display: flex;
    justify-content: center;
    margin-bottom: 2rem;
    gap: 1rem;
}

.tab-button {
    background: rgba(255,255,255,0.1);
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 50px;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.tab-button:hover {
    background: rgba(255,255,255,0.2);
    transform: translateY(-2px);
}

.tab-button.active {
    background: linear-gradient(45deg, var(--primary), var(--accent));
    box-shadow: 0 4px 15px rgba(110, 72, 170, 0.4);
}

.tab-content {
    display: none;
    animation: fadeIn 0.5s ease-out;
}

.tab-content.active {
    display: block;
}

.panel {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem auto;
    max-width: 900px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

.panel-header {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.panel-header svg {
    width: 1.5rem;
    height: 1.5rem;
}

button {
    background: linear-gradient(45deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 50px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(110, 72, 170, 0.3);
    margin: 0.5rem 0;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(110, 72, 170, 0.4);
}

button:active {
    transform: translateY(0);
}

button.secondary {
    background: rgba(255,255,255,0.1);
}

button.secondary:hover {
    background: rgba(255,255,255,0.2);
}

textarea, input[type="text"] {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    border-radius: 8px;
    padding: 0.8rem;
    width: 100%;
    margin-bottom: 1rem;
}

textarea:focus, input[type="text"]:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(71, 118, 230, 0.3);
}

.output-box {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    border-left: 4px solid var(--accent);
}

.output-label {
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: block;
    color: #ddd;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.slide-in {
    animation: slideIn 0.5s ease-out forwards;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.file-upload {
    border: 2px dashed rgba(255,255,255,0.3);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.file-upload:hover {
    border-color: var(--accent);
    background: rgba(71, 118, 230, 0.1);
}

.progress-bar {
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    margin-top: 1rem;
    overflow: hidden;
}

.progress {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    width: 0%;
    transition: width 0.3s ease;
}
'''

with gr.Blocks(css=css) as demo:
    gr.Markdown("""
    <div class='header'>
        <span style="font-size:1.2em">🔬</span> AI Research Companion 
        <span style="font-size:1.2em">🧠</span>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("📄 Upload PDF", id="upload"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("""<div class="panel-header">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    Document Processing
                </div>""")
                
                with gr.Column(elem_classes=["file-upload"]):
                    file_upload = gr.File(
                        file_types=['.pdf'], 
                        label="Drag & Drop PDF or Click to Browse",
                        elem_classes=["upload-box"]
                    )
                    upload_btn = gr.Button("Process Document", variant="primary")
                    status = gr.Textbox(label="Processing Status", interactive=False)
                    gr.Markdown("<div class='progress-bar'><div class='progress'></div></div>")

        with gr.TabItem("❓ Ask Questions", id="qa"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("""<div class="panel-header">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Research Q&A
                </div>""")
                
                question = gr.Textbox(
                    placeholder="Type your research question here...", 
                    label="Your Question",
                    lines=3
                )
                ask_btn = gr.Button("Get Answer", variant="primary")
                
                with gr.Column(elem_classes=["output-box"]):
                    gr.Markdown("<div class='output-label'>Answer</div>")
                    answer = gr.Textbox(show_label=False, lines=6, interactive=False)
                
                with gr.Column(elem_classes=["output-box"]):
                    gr.Markdown("<div class='output-label'>Source References</div>")
                    citations = gr.Textbox(show_label=False, lines=4, interactive=False)

        with gr.TabItem("✍️ Summarize", id="summary"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("""<div class="panel-header">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7" />
                    </svg>
                    Document Summary
                </div>""")
                
                summary_btn = gr.Button("Generate Summary", variant="primary")
                
                with gr.Column(elem_classes=["output-box"]):
                    gr.Markdown("<div class='output-label'>Key Insights</div>")
                    summary_output = gr.Textbox(show_label=False, lines=8, interactive=False)

        with gr.TabItem("🔍 Similar Papers", id="papers"):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("""<div class="panel-header">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    Related Research
                </div>""")
                
                similar_btn = gr.Button("Find Similar Papers", variant="primary")
                
                with gr.Column(elem_classes=["output-box"]):
                    gr.Markdown("<div class='output-label'>Recommended Papers</div>")
                    similar_output = gr.Textbox(show_label=False, lines=8, interactive=False)

    # Event handlers
    upload_btn.click(upload_pdf, inputs=file_upload, outputs=status)
    ask_btn.click(ask_question, inputs=question, outputs=[answer, citations])
    summary_btn.click(summarize_pdf, outputs=summary_output)
    similar_btn.click(find_similar_papers, outputs=similar_output)

if __name__ == "__main__":
    demo.launch()

           



