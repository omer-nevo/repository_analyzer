#  Repository Analyzer

A Python-based repository analysis tool leveraging OpenAI Assistant API for code insights, FAISS for vector storage, and Flask API for easy interaction.

---

## Installation Guide

### Clone the repository:
```bash
git clone https://github.com/omer-nevo/repository_analyzer.git
cd repository_analyzer
```

### Install the required dependencies
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Configuration
1. Rename `config.example.yaml` to `config.yaml`
2. Fill in your own API keys 

## Usage

### Run the Flask API
```bash
flask run --host=0.0.0.0 --port=5000
```

### Call API Endpoints
Test the API using **cURL** or **Postman**.

#### **List Repository Files**
```bash
curl -X GET "http://127.0.0.1:5000/list_files"
```

#### **Query Code Context**
```bash
curl -X POST "http://127.0.0.1:5000/query" -H "Content-Type: application/json" -d '{"query": "How does authentication work?"}'
```

---

## Design decisions

Chose the example project structure for clarity and maintainability.

API Framework - Flask:
- Familiar framework
- Uses REST API
- Simple and lightweight implementation

Vector Database - FAISS:
- Local storage
- Fast similarity search
- No external API calls (unlike Pinecone)
- Efficient performance

Embedding Model - text-embedding-3-small:
- Cost-effective
- Strong performance
- Outperforms text-embedding-ada-002 (per OpenAI)

Text Chunking - Chunk Size 500
- Good trade-off between completeness and efficiency 
- Compatible with FAISS
- Matches typical function sizes in code bases
Initially considered CintraAI Code Chunker
Time constraints led to fixed 500-token approach

Context-Aware Querying: Uses OpenAI’s embeddings to analyze code.

## Performance considerations

Uses batch processing for indexing repositories efficiently.
Implements rate limiting (AsyncLimiter) to handle OpenAI API calls.
Supports async execution (asyncio) for non-blocking operations.

## Future improvements

The current implementation does not work fluently yet and lacks some features required:

1. API Enhancements
The API is still in its basic form and requires further development.
Implement authentication and authorization mechanisms for security.
2. Smarter File Handling & Chunking
Currently, the repository does not handle various file types and encodings.
3. Assistant & Vector Database Integration
The Assistant can generate its own code snippets, but it is not yet integrated with the vector database to retrieve relevant code from indexed files.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)