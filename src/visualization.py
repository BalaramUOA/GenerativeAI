from sklearn.manifold import TSNE

def tsne_reduction(document_embeddings, n_components=3):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(document_embeddings)

