# Practical Experiment — Assignment 1: RAG System for Research Paper Q&A

**Subject:** Agentic AI  
**Students:** Harshit Arora & Aditi Jha

---

## A. Problem Statement

Research papers are dense, technical documents that are difficult to query using traditional keyword-based search methods. Large Language Models (LLMs) alone cannot process entire papers due to limited context windows, and they may hallucinate facts not present in the source material.

The objective of this practical is to design and implement a **Retrieval-Augmented Generation (RAG)** system that can accurately answer natural-language questions about a corpus of research papers by:

1. Ingesting PDF research papers and converting them into searchable vector representations.
2. Retrieving the most semantically relevant text chunks for a given user query using similarity search.
3. Augmenting an LLM prompt with the retrieved context so the model generates **accurate, grounded answers with source citations** — rather than relying solely on its parametric knowledge.

The system is built as both a Jupyter Notebook pipeline and an interactive Streamlit web application.

---

## B. Solution

### Step 1: Install Dependencies

Install the required Python libraries — `langchain`, `langchain-community`, `langchain-groq`, `langchain-classic`, `faiss-cpu`, `sentence-transformers`, `pypdf`, and `streamlit` — using `pip install -r requirements.txt`.

### Step 2: Import Libraries

Import all necessary modules:
- `PyPDFLoader` (for loading PDF files)
- `RecursiveCharacterTextSplitter` (for splitting text into chunks)
- `HuggingFaceEmbeddings` (for generating vector embeddings)
- `FAISS` (for vector storage and similarity search)
- `ChatGroq` (for LLM inference via Groq API)
- `RetrievalQA` and `PromptTemplate` (for building the RAG chain)

### Step 3: Configure the LLM API Key

Set the Groq API key as an environment variable. The system uses **Groq's cloud-hosted Llama 3.3 70B Versatile** model for answer generation.

### Step 4: Load PDF Documents

Load 5 research papers from the `research_papers/` directory using `PyPDFLoader`:

| # | Paper |
|---|-------|
| 1 | Attention Is All You Need (Transformer) |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers |
| 3 | Sentence-BERT: Sentence Embeddings using Siamese BERT |
| 4 | RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP |
| 5 | Recent advances in AI/NLP research |

Each PDF is parsed page-by-page, and all pages are collected into a single document list.

### Step 5: Split Text into Chunks

Use `RecursiveCharacterTextSplitter` to divide the loaded documents into smaller, overlapping chunks:

- **Chunk size:** 500 characters (~1 paragraph)
- **Chunk overlap:** 100 characters (prevents information loss at boundaries)
- **Separators:** `\n\n` → `\n` → ` ` → `""` (respects natural text boundaries)

### Step 6: Generate Embeddings

Load the `sentence-transformers/all-MiniLM-L6-v2` embedding model to convert each text chunk into a **384-dimensional vector**. Embeddings are normalized for consistent cosine similarity scoring.

### Step 7: Create FAISS Vector Store

Store all chunk embeddings in a **FAISS (Facebook AI Similarity Search)** index using Flat L2 (exact search). The index is persisted to disk at `faiss_index/` for reuse without re-computation.

### Step 8: Initialize the LLM

Initialize the Groq-hosted **Llama 3.3 70B Versatile** model with a temperature of 0.3 for deterministic, focused answer generation.

### Step 9: Create the Prompt Template

Design a custom prompt that instructs the LLM to:
- Answer questions **only** based on the retrieved context from the research papers.
- Cite which paper(s) are referenced in the answer.
- Acknowledge when insufficient information is available rather than hallucinate.

### Step 10: Build the RetrievalQA Chain

Combine the FAISS retriever (top-k = 10 similarity search) with the Groq LLM into a LangChain `RetrievalQA` chain using the "stuff" strategy (all retrieved chunks are stuffed into a single prompt).

### Step 11: Query the RAG System

Ask natural-language questions to the pipeline. For each query, the system:
1. Embeds the question using MiniLM.
2. Searches the FAISS index for the top-10 most similar chunks.
3. Passes the retrieved chunks + question to the LLM.
4. Returns a generated answer with source citations (paper name + page number).

Sample queries tested:
- *"What is the Transformer architecture and its key components?"*
- *"What is BERT and how does it differ from previous models?"*
- *"Explain the attention mechanism in neural networks."*
- *"What is Retrieval-Augmented Generation (RAG) and how does it work?"*
- *"How does Sentence-BERT generate sentence embeddings?"*

### Step 12: Run the Streamlit Web Application

Launch the interactive web UI using `streamlit run app.py`. The app provides:
- A premium dark-themed interface with a hero header.
- Real-time statistics (papers loaded, pages, chunks, model name).
- A text input for asking questions.
- Styled answer cards with response time and expandable source documents.

---

## C. Result

> **Screenshot 1 (Notebook — Pipeline Output):**
> A screenshot of the Jupyter Notebook output showing the successful execution of all pipeline steps — number of PDFs loaded (5 papers), total pages parsed, total chunks created, FAISS vector store creation time, and the confirmation messages for embedding model loading and RetrievalQA chain initialization.

> **Screenshot 2 (Notebook — Query Output):**
> A screenshot of the notebook output for a test query (e.g., *"What is the Transformer architecture and its key components?"*). The output displays the question, the LLM-generated answer with citations to specific papers, the response time, and a list of source documents with paper names, page numbers, and content previews.

> **Screenshot 3 (Streamlit App — Main Interface):**
> A screenshot of the Streamlit web application showing the dark-themed hero header ("📄 Research Paper Q&A"), the four statistics cards (Papers: 5, Pages count, Chunks count, Model: Llama 3.3 70B), the search input field, and the "Search & Answer" button.

> **Screenshot 4 (Streamlit App — Query Result):**
> A screenshot of the Streamlit app after submitting a question. The output shows the question badge, the generated answer with paper citations, the response time and source chunk count in the meta line, and the expandable "View Source Documents" section listing each retrieved chunk with its paper name, page number, and content preview.
