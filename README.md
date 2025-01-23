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

## Performance considerations

## Future improvements

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)