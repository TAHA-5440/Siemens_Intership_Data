# ==========================
# COMPLETE FIX: Full Mock of Distributed Module
# ==========================
import os
import sys
import types

print("=" * 60)
print("APPLYING COMPLETE DISTRIBUTED TRAINING FIX")
print("=" * 60)

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = "/media/jetson/lib/huggingface_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Force early imports
import sklearn
import numpy
import torch

print("✅ Pre-loaded dependencies")

# ==========================
# Create COMPLETE mock _distributed_c10d module
# ==========================
mock_distributed = types.ModuleType('torch._C._distributed_c10d')

# Mock all classes and constants that might be imported
class MockProcessGroup:
    pass

class MockWork:
    def wait(self):
        pass

class MockAllreduceOptions:
    pass

class MockAllreduceCoalescedOptions:
    pass

class MockBroadcastOptions:
    pass

class MockAllgatherOptions:
    pass

class MockReduceOptions:
    pass

class MockScatterOptions:
    pass

class MockGatherOptions:
    pass

class MockAllToAllOptions:
    pass

class MockBarrierOptions:
    pass

# Mock ReduceOp enum
class MockReduceOp:
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    PREMUL_SUM = 7

# Add all mocked items to the module
mock_distributed.ProcessGroup = MockProcessGroup
mock_distributed.Work = MockWork
mock_distributed.AllreduceOptions = MockAllreduceOptions
mock_distributed.AllreduceCoalescedOptions = MockAllreduceCoalescedOptions
mock_distributed.BroadcastOptions = MockBroadcastOptions
mock_distributed.AllgatherOptions = MockAllgatherOptions
mock_distributed.ReduceOptions = MockReduceOptions
mock_distributed.ScatterOptions = MockScatterOptions
mock_distributed.GatherOptions = MockGatherOptions
mock_distributed.AllToAllOptions = MockAllToAllOptions
mock_distributed.BarrierOptions = MockBarrierOptions
mock_distributed.ReduceOp = MockReduceOp

# Add constants
mock_distributed._DEFAULT_FIRST_BUCKET_BYTES = 1024
mock_distributed._DEFAULT_NO_TIMEOUT = -1

# Inject the complete mock module
sys.modules['torch._C._distributed_c10d'] = mock_distributed

print("✅ Complete distributed training mock injected")

# ==========================
# Patch transformers integration functions
# ==========================
def mock_is_deepspeed_zero3_enabled():
    return False

def mock_is_fsdp_managed_module(module):
    return False

# Import and patch
import transformers.integrations.deepspeed
import transformers.integrations.fsdp

transformers.integrations.deepspeed.is_deepspeed_zero3_enabled = mock_is_deepspeed_zero3_enabled
transformers.integrations.fsdp.is_fsdp_managed_module = mock_is_fsdp_managed_module

print("✅ Transformers integration functions patched")

# ==========================
# Now import everything else
# ==========================
import streamlit as st
from docx import Document
import PyPDF2
from typing import List
import traceback
import gc

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# ==========================
# GPU Configuration
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==========================
# Custom Embeddings Class (GPU-enabled)
# ==========================
class CustomSentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings using SentenceTransformer with GPU support"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model on GPU"""
        print(f"Loading embedding model: {model_name} on {DEVICE}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        print(f"✅ Embedding model loaded successfully on {DEVICE}!")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            device=DEVICE,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            device=DEVICE
        )
        return embedding.tolist()

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Tesla Assistant",
    page_icon="🚗",
    layout="wide"
)

# ==========================
# Initialize Session State
# ==========================
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None

# ==========================
# Load Document Function
# ==========================
def load_document(uploaded_file):
    """Extract text from PDF or DOCX"""
    text = ""

    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text

# ==========================
# Initialize Models (Session State) - GPU Enabled
# ==========================
def get_embedding_model():
    """Get or initialize embedding model on GPU"""
    if st.session_state.embedding_model is None:
        print("🔄 Loading embedding model on GPU...")
        st.session_state.embedding_model = CustomSentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    return st.session_state.embedding_model

def get_chat_model():
    """Get or initialize chat model on GPU"""
    if st.session_state.chat_model is None:
        print("🔄 Loading LLM model on GPU...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            model_id ="Qwen/Qwen2.5-0.5B-Instruct"
            
            print("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            print("  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
                )
            
            if DEVICE == "cuda":
                print("  Moving model to GPU...")
                model = model.to("cuda")
            
            print("  Creating text generation pipeline...")
            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=80,              # Shorter responses
                min_new_tokens=5,
                temperature=0.2,                # More deterministic
                top_p=0.85,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                return_full_text=False,
                device=0 if DEVICE == "cuda" else -1,
                framework="pt",
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            
            st.session_state.chat_model = llm
            print(f"✅ LLM loaded successfully on {DEVICE}!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(traceback.format_exc())
            raise
            
    return st.session_state.chat_model

# ==========================
# Clear Models Function
# ==========================
def clear_models():
    """Clear models from memory"""
    st.session_state.chat_model = None
    st.session_state.embedding_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🗑️ GPU memory cleared!")

    print("🗑️ Models cleared from memory!")

# ==========================
# Index Document
# ==========================
def index_document(uploaded_file, chunk_size, chunk_overlap):
    """Index uploaded document into vector store"""
    try:
        doc_text = load_document(uploaded_file)

        if not doc_text.strip():
            return False, "❌ No text extracted from document"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.create_documents([doc_text])

        embedding_model = get_embedding_model()

        st.session_state.vector_store = FAISS.from_documents(chunks, embedding_model)
        st.session_state.retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return True, f"✅ Successfully indexed {len(chunks)} chunks from {uploaded_file.name}"

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error during indexing:\n{error_details}")
        return False, f"❌ Error: {str(e)}"

# ==========================
# Format Retrieved Docs
# ==========================
def format_docs(retrieved_docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ==========================
# Generate Response
# ==========================
def generate_response(question):
    """Generate response using RAG"""
    if st.session_state.retriever is None:
        return "⚠️ Please upload and index a Tesla document first."

    try:
        chat_model = get_chat_model()

        # Detect if it's a greeting/small talk
        greetings = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon"]
        if question.lower().strip() in greetings:
            greeting_responses = {
                "hi": "Hello! I'm Tesla Assistant. How can I help you with your Tesla today?",
                "hello": "Hello! I'm here to help with Tesla-related questions. What would you like to know?",
                "hey": "Hey there! Ready to assist with your Tesla questions.",
                "how are you": "I'm functioning perfectly! How can I assist you with your Tesla today?",
                "good morning": "Good morning! How can I help you with your Tesla?",
                "good afternoon": "Good afternoon! What Tesla information can I help you with?"
            }
            return greeting_responses.get(question.lower().strip(), 
                "Hello! I'm Tesla Assistant. Ask me anything about your Tesla vehicle.")

        template = """<|im_start|>system
You are Tesla Assistant. You MUST respond in ENGLISH ONLY.

CRITICAL RULES:
1. Answer ONLY in English language
2. Use ONLY information from Context below
3. If Context lacks the answer, say: "I don't have that information in the document."
4. Maximum 2-3 sentences
5. Be direct and factual
6. NO Chinese, NO explanations about assumptions
<|im_end|>

<|im_start|>user
Context: {context}

Question: {question}
<|im_end|>

<|im_start|>assistant
Answer in English:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        parallel_chain = RunnableParallel({
            "context": st.session_state.retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()
        chain = parallel_chain | prompt | chat_model | parser

        response = chain.invoke(question)
        
        # POST-PROCESSING: Clean the response
        response = response.strip()
        
        # Remove Chinese characters
        import re
        response = re.sub(r'[\u4e00-\u9fff]+', '', response)
        
        # Remove unwanted phrases
        unwanted = ["Note:", "I apologize", "For more detailed", "please refer", 
                   "Safe travels", "If the user", "This response is based",
                   "(Hello", "What can I do", "你好"]
        for phrase in unwanted:
            response = response.replace(phrase, "")
        
        # Remove parentheses content
        response = re.sub(r'\$[^)]*\$', '', response)
        
        # Clean whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Limit to 3 sentences
        sentences = [s.strip() + '.' for s in response.split('.') if s.strip()]
        response = ' '.join(sentences[:3])
        
        if len(response) < 10:
            return "I don't have that information in the uploaded document."
        
        return response

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error generating response:\n{error_details}")
        return f"❌ Error generating response: {str(e)}"

# ==========================
# Main UI
# ==========================
def main():
    st.title("🚗🤖 Tesla Assistant")
    st.markdown("*Your AI-powered guide to Tesla vehicles*")

    if DEVICE == "cuda":
        st.success(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ Running on CPU (GPU not available)")

    with st.sidebar:
        st.header("📄 Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a Tesla PDF or DOCX file",
            type=["pdf", "docx"],
            help="Upload Tesla documentation"
        )

        st.markdown("---")

        st.subheader("⚙️ Indexing Settings")
        chunk_size = st.slider("Chunk Size", 100, 1000, 400, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 10)

        if st.button("🔄 Index Document", type="primary", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Indexing document on GPU..."):
                    success, message = index_document(uploaded_file, chunk_size, chunk_overlap)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("⚠️ Please upload a file first")

        st.markdown("---")

        st.subheader("🧹 Memory Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("🔥 Clear Models", use_container_width=True):
                clear_models()
                st.success("✅ Models cleared!")

        st.markdown("---")

        st.subheader("📊 Status")

        if DEVICE == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            st.info(f"GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

        if st.session_state.retriever is not None:
            st.success("✅ Document indexed")
        else:
            st.info("ℹ️ No document indexed")

        if st.session_state.chat_model is not None:
            st.success("✅ LLM loaded")
        else:
            st.info("ℹ️ LLM not loaded")

    st.header("💬 Chat")

    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    question = st.chat_input("Ask about your Tesla...")

    if question:
        st.session_state.chat_history.append((question, ""))

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking on GPU..."):
                response = generate_response(question)
                st.write(response)

        st.session_state.chat_history[-1] = (question, response)
        st.rerun()

if __name__ == "__main__":
    main()
