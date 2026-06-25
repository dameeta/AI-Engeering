from rank_bm25 import BM25Okapi
import pickle

chunks = pickle.load(
  open("data/chunks.pkl","rb")
  
)

tokenized = [
  c.split()
  for c in chunks
]

bm25 = BM25Okapi(tokenized)

pickle.dump(
  bm25,open("data/bm25.pkl","wb")
)