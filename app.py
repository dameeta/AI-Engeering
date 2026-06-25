from fastapi import FastAPI
from search import hybrid_search
from rerank import rerank

app = FastAPI()

@app.get("/search")
def search(query:str):
  docs=hybrid_search(query)
  return docs