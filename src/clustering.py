from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def kmeans_clustering(document_embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(document_embeddings)

def lda_topic_modeling(documents, num_clusters):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=num_clusters, random_state=42)
    lda.fit(X)
    return lda, vectorizer

