from data_loader import load_data
from embedding import embed_documents
from clustering import kmeans_clustering, lda_topic_modeling
from visualization import tsne_reduction
from summarization import summarize_documents
from sentiment_analysis import analyze_sentiment
from nps_analysis import analyze_nps_scores
from utils import download_nltk_data

def main():
    # Download necessary NLTK data
    download_nltk_data()

    # Load data
    df = load_data('/content/drive/MyDrive/survey_insurance.csv')
    documents = df['responses']
    nps_scores = df['nps']

    # Embed documents
    document_embeddings = embed_documents(documents)

    # Perform KMeans clustering
    num_clusters = 7
    # just hardcoded number of cluster. It can be deceided as per the elbow curve or Silhouette score
    cluster_ids = kmeans_clustering(document_embeddings, num_clusters)

    # Perform LDA topic modeling
    lda_model, vectorizer = lda_topic_modeling(documents, num_clusters)

    # Perform TSNE reduction
    # TSNE not required for this excersice
    #vectors_tsne = tsne_reduction(document_embeddings)

    # Summarize documents
    summary = summarize_documents(documents)

    # Analyze sentiment
    # sentiment analysis not required for this excercise

    # Analyze NPS scores
    nps_sentiments = analyze_nps_scores(nps_scores)

    # Perform further analysis and visualization
    # ...

    # Display results
    # ...

if __name__ == "__main__":
    main()

