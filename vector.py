import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Ensure you have your OpenAI API key set in the environment
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'

def create_embeddings(documents):
    # Initialize the OpenAI embeddings model
    embeddings_model = OpenAIEmbeddings()

    # Create embeddings for each document
    embeddings = [embeddings_model.embed_text(doc) for doc in documents]
    return embeddings

def store_embeddings(documents, embeddings):
    # Create a Chroma vector store and add documents with their embeddings
    vector_store = Chroma()
    vector_store.add_texts(documents, embeddings)
    return vector_store

def main():
    # Sample text data
    documents = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

    # Create embeddings for the documents
    embeddings = create_embeddings(documents)

    # Store embeddings in a vector store
    vector_store = store_embeddings(documents, embeddings)

    # Example: Retrieve a document by its embedding
    query = "Find the first document."
    query_embedding = OpenAIEmbeddings().embed_text(query)
    results = vector_store.similarity_search(query_embedding, k=1)
    
    print("Most similar document to query:", results[0])

if __name__ == "__main__":
    main()
