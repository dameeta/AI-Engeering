import faiss
import pickle
import numpy as np

embeddings = pickle.load(
  open("data/embeddings.pkl","rb")
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
  dimension
)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index,"data/faiss_index")

print("FAISS index created")