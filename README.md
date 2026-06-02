# AI RAG Agent

An intelligent Retrieval Augmented Generation (RAG) agent built with Python. This agent combines document retrieval with large language models to provide accurate, context-aware responses.

## Features

- **Document Retrieval**: Fast and efficient document indexing and retrieval
- **LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- **Semantic Search**: Vector-based similarity search using embeddings
- **Query Processing**: Intelligent query understanding and context building
- **Caching**: Efficient caching to reduce API calls

## Project Structure

```
.
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── rag_agent.py          # Main RAG agent class
│   │   └── config.py             # Configuration management
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── document_store.py     # Document storage and indexing
│   │   └── embeddings.py         # Embedding generation
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_client.py         # LLM client wrapper
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py    # Text preprocessing utilities
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_retrieval.py
│   └── test_llm.py
├── examples/
│   ├── basic_example.py          # Basic usage example
│   └── advanced_example.py       # Advanced usage example
├── requirements.txt
├── .env.example
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SamSegz/AI-RAG-Agent.git
cd AI-RAG-Agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Quick Start

```python
from src.agent.rag_agent import RAGAgent

# Initialize the agent
agent = RAGAgent(
    llm_provider="openai",
    embedding_model="text-embedding-3-small"
)

# Add documents
agent.add_documents(["path/to/documents"])

# Query the agent
response = agent.query("What is this document about?")
print(response)
```

## Configuration

See `.env.example` for all available configuration options:
- LLM API keys
- Embedding model selection
- Vector store settings
- Cache configuration

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open an issue on GitHub.
