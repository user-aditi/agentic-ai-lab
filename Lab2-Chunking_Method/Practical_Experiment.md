# Practical Experiment — Lab 2: Text Chunking Methods

**Subject:** Agentic AI  
**Students:** Harshit Arora & Aditi Jha

---

## A. Problem Statement

Large Language Models have limited context windows and cannot process entire documents at once. To effectively use LLMs for tasks like question answering, summarization, or retrieval-augmented generation, long documents must be split into smaller, meaningful pieces — a process known as **chunking** or **text splitting**.

The objective of this practical is to explore and compare **5 levels of text splitting strategies**, progressing from simple to advanced:

1. **Character Splitting** — Naive fixed-size chunking.
2. **Recursive Character Text Splitting** — Structure-aware chunking using hierarchical separators.
3. **Document-Specific Splitting** — Format-aware chunking for Markdown, Python, and JavaScript.
4. **Semantic Chunking** — Meaning-based chunking using embedding similarity.
5. **Agentic Chunking** — LLM-driven intelligent chunking at natural topic boundaries.

Each method is demonstrated using LangChain's text splitting utilities, and the experiment highlights the trade-offs between simplicity, structural awareness, and semantic understanding.

---

## B. Solution

### Step 1: Install Dependencies

Install the required libraries — `langchain-text-splitters`, `langchain_experimental`, and `langchain_openai` — using pip on Google Colab.

### Step 2: Level 1 — Character Splitting

Define a sample text string and split it into fixed-size chunks of 35 characters using a manual loop. Then, replicate the same using LangChain's `CharacterTextSplitter` with `chunk_size=35`, `chunk_overlap=0`, and an empty string separator. The output produces LangChain `Document` objects.

Demonstrate **chunk overlap** by setting `chunk_overlap=4`, showing how the tail of one chunk overlaps with the head of the next chunk to preserve continuity at boundaries.

### Step 3: Level 2 — Recursive Character Text Splitting

Load a longer multi-paragraph essay text (Paul Graham's essay on superlinear returns). Use `RecursiveCharacterTextSplitter` with `chunk_size=65` and `chunk_overlap=0`. This splitter uses a hierarchy of separators (`\n\n` → `\n` → ` ` → `""`) to respect natural text boundaries.

Observe that chunks now tend to end at paragraph breaks (periods followed by double newlines), unlike the rigid character-based splitting. Also demonstrate adding custom metadata (e.g., `source_file`, `chunk_no`) to each document chunk.

Increase chunk size to 450 to show how the splitter naturally "snaps" to paragraph boundaries for cleaner splits.

### Step 4: Level 3 — Document-Specific Splitting

#### Markdown Splitting
Use `MarkdownTextSplitter` with `chunk_size=40` on a sample Markdown document with headings (`#`, `##`, `###`). The splitter prioritizes splitting at heading boundaries, keeping sections together.

#### Python Code Splitting
Use `PythonCodeTextSplitter` with `chunk_size=100` on a Python code snippet containing a class definition and a for loop. The splitter uses Python-specific separators (`\nclass`, `\ndef`, `\n\n`, `\n`) to keep logical code blocks intact.

#### JavaScript Code Splitting
Use `RecursiveCharacterTextSplitter.from_language(Language.JS)` with `chunk_size=65` on a JavaScript snippet. The splitter uses JS-specific separators (`\nfunction`, `\nlet`, `\nvar`, etc.) to split at meaningful code boundaries.

### Step 5: Level 4 — Semantic Chunking

Configure the OpenAI API key and use `SemanticChunker` from `langchain_experimental` with OpenAI embeddings. Apply it to a Tesla Q3 earnings report text with three distinct sections (Q3 Results, Model Y Performance, Production Challenges).

The semantic chunker uses:
- **Embedding model:** `text-embedding-3-small` (OpenAI)
- **Breakpoint threshold type:** Percentile
- **Breakpoint threshold amount:** 70

Additionally, manually generate embeddings for individual sentences and compute a **cosine similarity matrix** to visualize how semantic similarity between adjacent sentences determines where to split. Sentences within the same topic have high similarity scores, while topic transitions show lower similarity — indicating natural split points.

### Step 6: Level 5 — Agentic Chunking

Use an LLM (`gpt-5-nano` via `ChatOpenAI`) to intelligently chunk the same Tesla earnings text. The LLM is prompted with specific rules:
- Each chunk should be ~200 characters or less.
- Split at natural topic boundaries.
- Keep related information together.
- Insert `<<<SPLIT>>>` markers at split points.

The LLM's response is then parsed by splitting on the `<<<SPLIT>>>` markers. The resulting chunks are cleaned (whitespace removed, empty chunks filtered) and displayed with character counts, demonstrating how an AI agent can identify topic boundaries more naturally than rule-based approaches.

---

## C. Result

> **Screenshot 1 (Level 1 — Character Splitting Output):**
> A screenshot showing the output of fixed-size character splitting — the text split into chunks of 35 characters each as a list of strings, followed by the LangChain `Document` objects with the same chunks. A second output shows the overlap version where 4 characters of the tail of each chunk match the head of the next chunk.

> **Screenshot 2 (Level 2 — Recursive Character Splitting Output):**
> A screenshot showing the essay text split into document chunks using `RecursiveCharacterTextSplitter`. The output displays a list of `Document` objects with metadata (`source_file`, `chunk_no`). The chunks align with paragraph boundaries, with many ending at natural sentence/paragraph breaks rather than mid-word.

> **Screenshot 3 (Level 3 — Document-Specific Splitting Output):**
> A screenshot showing three separate outputs — (a) Markdown text split at heading boundaries, (b) Python code split at class/function boundaries, and (c) JavaScript code split at function/variable declaration boundaries. Each output displays `Document` objects demonstrating format-aware chunking.

> **Screenshot 4 (Level 4 — Semantic Chunking Output):**
> A screenshot showing two outputs — (a) the semantic chunker results displaying chunks grouped by meaning (e.g., all Q3 revenue sentences together, Model Y sentences together, production challenge sentences together), and (b) the cosine similarity scores between adjacent sentences showing high similarity within topics (e.g., ~0.8) and lower similarity at topic transitions (e.g., ~0.5), demonstrating how embeddings detect semantic boundaries.

> **Screenshot 5 (Level 5 — Agentic Chunking Output):**
> A screenshot showing the LLM-generated chunking results. The output displays "Asking AI to chunk the text..." followed by the marked text with `<<<SPLIT>>>` markers, and then the final clean chunks with character counts. The chunks correspond to the three logical topics (Q3 Results, Model Y Performance, Production Challenges), demonstrating intelligent topic-boundary detection by the LLM.
