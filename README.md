# B5 RAG Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed for exact keyword retrieval from academic papers, demonstrating extraction and question-answering on reinforcement learning literature. The pipeline evaluates a base Large Language Model's raw knowledge against context-augmented responses using a customized TF-IDF and FAISS retrieval stack.

## Overview

The system processes multi-column academic PDF layouts, extracts text efficiently, applies sliding window chunking, and vectorizes the resulting text corpus using TF-IDF. This methodology is explicitly used over dense semantic embeddings due to its prioritization of exact keyword matching, which is highly effective for querying precise technical specifications (for example: training steps, dataset names, model variants). 

The generated context is subsequently passed to the `llama-3.3-70b-versatile` model via the Groq API for precise answer generation based exclusively on the retrieved chunks.

## Features

- **Robust PDF Extraction**: Utilizes `pdfplumber` to accurately read multi-column academic formats, avoiding common parsing errors inherent in standard PDF readers.
- **Sliding Window Chunking**: Maintains context stability across split segments by ensuring facts straddling chunk boundaries remain whole.
- **Keyword-Targeted Retrieval**: Deploys `TfidfVectorizer` (with n-grams) paired with `faiss.IndexFlatL2` for performant exact-match retrieval, ensuring high precision for academic inquiries.
- **Comparative Evaluation**: Executes a side-by-side comparison between the model's unassisted memory-based responses and its RAG-augmented answers.

## Setup Requirements

### Prerequisites

- Python 3.8 or higher
- An active Groq API Key

### Installation

1. Clone this repository to your local machine.
2. Install the necessary Python packages:
   ```bash
   pip install pdfplumber scikit-learn faiss-cpu groq numpy
   ```
3. Set your Groq API key as an environment variable in your terminal session:
   ```bash
   export YOUR_API_KEY="gsk_..."
   ```

## Usage

After configuring your environment variable, execute the pipeline script directly:

```bash
python b5_rag_pipeline.py
```

The script will output:
1. Progress logs mapping extraction, chunking, and FAISS indexing.
2. The model's unassisted response (without RAG).
3. The RAG-augmented response using specific chunks.
4. An overview of the top relevant chunks retrieved by FAISS.
