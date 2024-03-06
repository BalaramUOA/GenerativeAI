from transformers import pipeline

def summarize_documents(documents):
    summarizer = pipeline("summarization")
    return summarizer('. '.join(documents), max_length=150, min_length=30, do_sample=False)[0]['summary_text']

