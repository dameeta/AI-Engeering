import faiss
import pickle
import numpy as np

from sentence_transformers import (SentenceTransformer)

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss_index")
chunks=pickle.load(open("data/chunks.pkl","rb"))
bm25=pickle.load(open("data/bm25.pkl","rb"))

def hybrid_search(query, top_k=10):
  q_embed = model.encode([query])
  
  D, I =index.search (
    np.array(q_embed).astype("float32"),top_k
    )
  bm25_scores = bm25.get_scores(query.split())
  
  results = []
  for idx in I[0]:
    vector_score = float(
      D[0][
        list(I[0]).index(idx)]
           
    )
    bm25_score = float(
      bm25_scores[idx]
    )
    final_score=(
      0.6*vector_score +
      0.4 * bm25_score
    )
    results.append(
      {
        "text" :chunks[idx],
        "score" : final_score
      }
    )
  results.sort(
    key=lambda x:x["score"],
    reverse=True
  )
  
  return results