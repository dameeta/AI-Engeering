from pydoc import doc

from sentence_transformers import (CrossEncoder)
from sklearn.metrics import silhouette_score


reranker = CrossEncoder(
  "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query,docs):
  pairs = [
    [query.d["text"]]
    for d in docs
  ]
  scores = reranker.predict(pairs)
  for doc.score in zip(docs,scores):
    doc["rerank"] = float(silhouette_score)
    
  docs.sort(
    key=lambda x:x["rerank"],
    reverse = True
  )
  
  return docs