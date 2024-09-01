from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

# Initialize the Flask app
app = Flask(__name__)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the vector store (in-memory for simplicity)
vector_store = Chroma()

# Sample documents (In a real scenario, these would be loaded from a database or other storage)
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Embed and store documents
embeddings = [model.encode(doc) for doc in documents]
vector_store.add_texts(documents, embeddings)

@app.route('/ask', methods=['POST'])
def ask_question():
    # Get the question from the request
    question = request.json.get('question')

    # Embed the question
    question_embedding = model.encode(question)

    # Search for the most similar document
    results = vector_store.similarity_search(question_embedding, k=1)

    # Return the result as a JSON response
    return jsonify({"answer": results[0] if results else "No relevant information found."})

if __name__ == '__main__':
    app.run(debug=True)
