# Practical Experiment — Lab 3: Multi-Modal RAG Pipeline

**Subject:** Agentic AI  
**Students:** Harshit Arora & Aditi Jha

---

## A. Problem Statement

Traditional RAG (Retrieval-Augmented Generation) systems treat PDF documents as plain text, ignoring rich structural elements such as tables, images, diagrams, and section headings. This leads to loss of critical information — for example, data in tables is jumbled into unstructured text, and visual content like architecture diagrams is completely discarded.

The objective of this practical is to build a **Multi-Modal RAG pipeline** that:

1. Parses a PDF document into structured elements (titles, paragraphs, tables, images) using the `Unstructured.io` library.
2. Groups these elements into intelligent, title-based chunks that preserve document structure.
3. Uses a vision-capable LLM (GPT-4o) to generate AI-enhanced searchable summaries for chunks containing tables and images.
4. Stores the enriched chunks in a ChromaDB vector database with OpenAI embeddings.
5. Retrieves relevant multi-modal chunks for a user query and generates a grounded answer using the retrieved text, tables, and images.

The pipeline is demonstrated on the "Attention Is All You Need" (Transformer) research paper.

---

## B. Solution

### Step 1: Install System Dependencies

Install system-level tools required for PDF processing on Google Colab:
- **Poppler** (`poppler-utils`) — Extracts text, images, and metadata from PDFs.
- **Tesseract** (`tesseract-ocr`) — OCR engine for reading text from scanned documents and images.
- **libmagic** (`libmagic-dev`) — Detects file types by analyzing content rather than extension.

### Step 2: Install Python Libraries

Install Python packages using pip:
- `unstructured[all-docs]` — Document parsing and partitioning.
- `langchain_chroma` — ChromaDB vector store integration.
- `langchain`, `langchain-community`, `langchain-openai` — RAG framework and OpenAI LLM/embedding integration.

### Step 3: Configure OpenAI API Key

Set the OpenAI API key from Google Colab's user data secrets as an environment variable.

### Step 4: Import Libraries

Import all necessary modules:
- `partition_pdf`, `chunk_by_title` from `unstructured` for document parsing and chunking.
- `Document`, `ChatOpenAI`, `OpenAIEmbeddings`, `Chroma`, `HumanMessage` from LangChain for RAG components.

### Step 5: Partition the PDF Document

Use `partition_pdf()` to break the PDF ("Attention Is All You Need") into structured elements with the following settings:
- **Strategy:** `hi_res` (most accurate extraction method).
- **Infer table structure:** `True` (preserves tables as structured HTML).
- **Extract images:** `True` (stores images as base64 data).

Inspect the extracted elements — identify their types (Title, NarrativeText, Table, Image, etc.), view individual element dictionaries, and separately gather all images and tables for analysis.

### Step 6: Create Title-Based Chunks

Use `chunk_by_title()` to intelligently group elements into logical chunks based on section headings:
- **Max characters:** 3000 (hard limit per chunk).
- **New chunk after:** 2400 characters (soft target for starting a new chunk).
- **Combine under:** 500 characters (merge tiny chunks with neighbors).

Inspect the created chunks, their types, and their original elements metadata.

### Step 7: Separate Content Types and Generate AI Summaries

For each chunk, analyze its content types using `separate_content_types()` — extracting raw text, table HTML, and image base64 data separately.

For chunks containing tables or images, use **GPT-4o** (a vision-capable LLM) via `create_ai_enhanced_summary()` to generate comprehensive, searchable descriptions that cover:
- Key facts, numbers, and data points from text and tables.
- Main topics and concepts discussed.
- Questions the content could answer.
- Visual content analysis (charts, diagrams, patterns).
- Alternative search terms users might use.

Chunks with only text use the raw text as-is. Each processed chunk is stored as a LangChain `Document` with rich metadata (original text, table HTML, image base64).

### Step 8: Export Chunks to JSON

Export all processed chunks to a JSON file (`chunks_export.json`) with structured data including chunk ID, enhanced content, and original metadata for inspection and debugging.

### Step 9: Create ChromaDB Vector Store

Store all processed chunks in a **ChromaDB** vector database using:
- **Embedding model:** OpenAI `text-embedding-3-small`.
- **Distance metric:** Cosine similarity (`hnsw:space: cosine`).
- **Persistence:** Saved to disk for reuse.

### Step 10: Run the Complete Ingestion Pipeline

Execute the full pipeline in one function call — Partition → Chunk → AI Summarize → Vector Store — to create a second, fully processed database.

### Step 11: Query and Generate Multi-Modal Answers

Query the vector store with a natural-language question (e.g., *"How many attention heads does the Transformer use, and what is the dimension of each head?"*). Retrieve the top-3 most relevant chunks.

Use `generate_final_answer()` with **GPT-4o** to generate the final answer by:
1. Building a prompt containing the raw text and table HTML from each retrieved chunk.
2. Appending any base64-encoded images from retrieved chunks as image inputs to GPT-4o's vision capability.
3. Sending the multi-modal message and returning the generated answer.

---

## C. Result

> **Screenshot 1 (Document Partitioning Output):**
> A screenshot of the notebook output showing the partitioning of the "Attention Is All You Need" PDF. The output displays "Partitioning document: /content/attention-is-all-you-need.pdf" followed by the total number of extracted elements. It also shows the set of unique element types found (e.g., Title, NarrativeText, Table, Image, Header, Footer, etc.), the number of images and tables extracted, and a sample element dictionary with its metadata.

> **Screenshot 2 (Chunking and AI Summary Processing Output):**
> A screenshot showing the chunk processing log. The output displays "Creating smart chunks..." with the total number of chunks created, followed by the AI summarization progress — "Processing chunk 1/N", "Types found: ['text']", "→ Using raw text (no tables/images)" for text-only chunks, and "→ Creating AI summary for mixed content..." with "→ AI summary created successfully" and a preview of the enhanced content for chunks containing tables or images.

> **Screenshot 3 (Vector Store and Query Result):**
> A screenshot showing the vector store creation ("Creating embeddings and storing in ChromaDB...") and the final query result. The output displays the question (*"How many attention heads does the Transformer use, and what is the dimension of each head?"*) followed by GPT-4o's generated answer — a clear, comprehensive response citing specific details from the Transformer paper (e.g., 8 attention heads, dimension of 64 per head), demonstrating that the multi-modal RAG pipeline successfully retrieves and synthesizes information from text, tables, and images.
