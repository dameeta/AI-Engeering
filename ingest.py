from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import pickle
model=SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,
  chunk_overlap=100
)
with open("data/documents.json","r") as f:
  documents = json.load(f)
all_chunks=[]
for doc in documents:
  chunks = splitter.split_text(doc["text"])
  all_chunks.extend(chunks)

embeddings=model.encode(all_chunks)
pickle.dump(
  all_chunks,
  open("data/chunks.pkl","wb")
  )
pickle.dump(
  embeddings,open("data/embeddings.pkl","wb")
)
