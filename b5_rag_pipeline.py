import os
import re
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from groq import Groq

PDF_PATH = "/Users/prithvirajsangramsinhpatil/genai-assignment/rlm.pdf"         
MAIN_PAPER_PAGES = (1, 38)    
CHUNK_SIZE = 400             
CHUNK_OVERLAP = 80           
TOP_K = 4                    


# ── Step 1: Extract text from PDF ─────────────────────────────────────────────
def extract_text(pdf_path, start_page, end_page):
    """
    Extract and lightly clean text from a page range using pdfplumber.
    Handles multi-column academic layout better than pypdf.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[start_page - 1 : end_page]:
            text = page.extract_text()
            if text:
                text = re.sub(r'-\n', '', text)       
                text = re.sub(r'\n+', ' ', text)      
                text = re.sub(r'\s{2,}', ' ', text)   
                full_text.append(text.strip())
    return ' '.join(full_text)


print("Step 1: Extracting text from PDF...")
raw_text = extract_text(PDF_PATH, *MAIN_PAPER_PAGES)
print(f"  Extracted {len(raw_text):,} characters from pages {MAIN_PAPER_PAGES[0]}-{MAIN_PAPER_PAGES[1]}")


# ── Step 2: Chunk the text ─────────────────────────────────────────────────────
def chunk_text(text, chunk_size, overlap):
    """
    Sliding window chunker.
    Overlap prevents facts that straddle chunk boundaries from being missed.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]


chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"\nStep 2: Chunking complete")
print(f"  {len(chunks)} chunks (size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars)")


# ── Step 3: Embed with TF-IDF ──────────────────────────────────────────────────
# TF-IDF produces sparse vectors that work well for exact keyword retrieval,
# which is appropriate here since our queries use precise technical terms
# (e.g. "training steps", "H100", "LongBenchPro").
# For semantic similarity on vague queries, swap for sentence-transformers.

print("\nStep 3: Building TF-IDF embeddings...")
vectorizer = TfidfVectorizer(
    max_features=4096,    # vocabulary size cap
    ngram_range=(1, 2),   # unigrams + bigrams to capture phrases
    sublinear_tf=True     # log-normalise term frequency
)
# Fit on the corpus so rare technical terms get high IDF weight
tfidf_matrix = vectorizer.fit_transform(chunks).toarray().astype(np.float32)
dimension = tfidf_matrix.shape[1]
print(f"  TF-IDF matrix: {tfidf_matrix.shape[0]} chunks x {dimension} features")


# ── Step 4: Build FAISS index ──────────────────────────────────────────────────
# IndexFlatL2: exact nearest-neighbour by L2 distance.
# No approximation -- correct for a corpus this small.
print("\nStep 4: Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(tfidf_matrix)
print(f"  Index built: {index.ntotal} vectors, dimension={dimension}")


# ── Step 5: Retrieval function ─────────────────────────────────────────────────
def retrieve(query, top_k=TOP_K):
    """
    Vectorise query with the same fitted TF-IDF, search FAISS.
    Returns (chunk, l2_distance) pairs. Lower L2 = more relevant.
    """
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(q_vec, top_k)
    return [(chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0])]


# ── Step 6: RAG vs no-RAG comparison ──────────────────────────────────────────
client = Groq(api_key=os.environ.get("YOUR_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

QUESTION = (
    "What specific training recipe did the authors use to create RLM-Qwen3-8B? "
    "Include the number of training samples, training steps, compute used, "
    "and what data source the trajectories came from."
)

# Without RAG
print("\n" + "=" * 65)
print("WITHOUT RAG  (model memory only)")
print("=" * 65)
print(f"Q: {QUESTION}\n")

no_rag_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
    temperature=0.2
)
print(no_rag_resp.choices[0].message.content)

# With RAG
print("\n" + "=" * 65)
print("WITH RAG  (TF-IDF + FAISS retrieval from rlm.pdf)")
print("=" * 65)

results = retrieve(QUESTION)
context = "\n\n".join(
    f"[Chunk {i+1}] {chunk}" for i, (chunk, _) in enumerate(results)
)

rag_prompt = f"""You are a precise research assistant. Answer the question using 
ONLY the provided context extracted from the paper. If the context does not 
contain enough information to answer fully, say so -- do not guess.

Context:
{context}

Question: {QUESTION}

Answer:"""

rag_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": rag_prompt}],
    temperature=0.2
)
print(rag_resp.choices[0].message.content)

# Show retrieved chunks
print("\n" + "=" * 65)
print(f"RETRIEVED CHUNKS  (top {TOP_K}, lower L2 = more relevant)")
print("=" * 65)
for i, (chunk, dist) in enumerate(results):
    print(f"\n[Chunk {i+1}]  L2 = {dist:.4f}")
    print(chunk)