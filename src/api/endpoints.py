import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from flask import Flask, request, jsonify
from src.core.vectorstore import VectorStore
from src.core.assistant import OpenAIAssistant
import asyncio

# Initialize Flask app
app = Flask(__name__)

# Initialize the Vector Store and OpenAI Assistant
vector_store = VectorStore()
assistant = OpenAIAssistant()


@app.route("/", methods=["GET"])
def root():
    """Check if API is running."""
    return jsonify({"message": "Repository Analyzer API is running"}), 200


@app.route("/search", methods=["POST"])
def search_vector_store():
    """
    Search for relevant results in the FAISS vector database.
    """
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query text is required"}), 400

        results = vector_store.search(query, top_k=3)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/ask-assistant", methods=["POST"])
def ask_assistant():
    """
    Queries OpenAI Assistant API with the given text and returns its response.
    """
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query text is required"}), 400

        response = asyncio.run(assistant.query(query))  # Run async function in sync Flask environment
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": f"Assistant query failed: {str(e)}"}), 500


@app.route("/index-repo", methods=["POST"])
def index_repository():
    """
    Indexes the repository files into the FAISS vector store.
    """
    try:
        asyncio.run(vector_store.index_repository_files())  # Run async function in sync Flask environment
        return jsonify({"message": "Repository successfully indexed"}), 200
    except Exception as e:
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
