# AI-Engeering
Steps to run this app
1. run python ingest.py
2. run python vector_store.py
3. run python search.py
4. run bm25_store.py
5. run rerank.py
6. uvicorn app:app --reload
7. Test http://localhost:8000/docs
8. search?query=vacation policy or leave policy it should display json data releavnt to leave
