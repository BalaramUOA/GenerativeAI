from sentence_transformers import SentenceTransformer

def embed_documents(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(documents)

