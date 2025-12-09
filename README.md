# 🚀 NexusRAG: Advanced Document Intelligence Platform

**NexusRAG** is a cutting-edge Retrieval-Augmented Generation (RAG) system that transforms how you interact with your documents. Have natural conversations with your knowledge base and extract insights with unprecedented accuracy.

## ✨ Core Features

### 🧠 Advanced Hybrid Retrieval
- **Dual-Engine Search**: Combines FAISS vector search (semantic) with BM25 (lexical) for superior retrieval quality
- **Cohere Reranking**: Implements state-of-the-art reranking to prioritize the most relevant context
- **Adaptive Chunking**: Intelligent document segmentation preserves context while optimizing for retrieval

### 📚 Universal Document Support
- **Multi-Format Processing**: Seamlessly handles PDF, Word, TXT, Markdown, and HTML files
- **Structure Preservation**: Maintains document structure and relationships during processing
- **Metadata Enrichment**: Automatically enhances documents with source tracking and attribution

### 💬 Intelligent Conversation
- **Claude AI Integration**: Powered by Anthropic's Claude for nuanced understanding and natural responses
- **Contextual Memory**: Maintains conversation history for coherent multi-turn interactions
- **Source Attribution**: Every answer includes references to specific source documents

### ⚡ Performance Optimizations
- **Async Architecture**: Built with asyncio for responsive performance even with large document collections
- **Efficient Indexing**: FAISS vector database enables millisecond retrieval from thousands of documents
- **Streaming Responses**: Real-time token-by-token display for immediate feedback

### 🛡️ Enterprise-Ready
- **Session Management**: Supports multiple concurrent users with isolated conversation contexts
- **Robust Error Handling**: Comprehensive logging and graceful failure recovery
- **Configurable Parameters**: Fine-tune retrieval settings, model behavior, and chunking strategies

## 🔧 Technical Architecture

```
nexusrag/
├── app/                    # Application layer
│   └── web_chatbot.py      # Streamlit web interface
├── src/                    # Core source code
│   ├── chains/             # LangChain chains
│   ├── chat_model/         # LLM integration
│   ├── embedding/          # Vector embeddings
│   ├── loaders/            # Document loaders
│   ├── memory/             # Conversation memory
│   ├── prompts/            # Prompt templates
│   ├── utils/              # Utility functions
│   ├── vectorstores/       # Vector store implementations
│   └── config.py           # Configuration
├── data/                   # Data directory
│   ├── documents/          # Document storage
│   ├── faiss_index/        # FAISS vector indices
│   └── long_term_memory/   # Persistent memory storage
└── requirements.txt        # Dependencies
```

## 🔍 How It Works

1. **Document Processing Pipeline**:
   - Documents are loaded using specialized loaders for each format
   - Text is extracted and split into optimally-sized chunks
   - Each chunk is embedded using state-of-the-art embedding models
   - Embeddings are indexed in FAISS for efficient vector search

2. **Hybrid Retrieval System**:
   - User queries are processed through both semantic and keyword search
   - Results are combined and reranked based on relevance
   - Top matches are selected as context for the language model

3. **Conversational AI Layer**:
   - Retrieved context is formatted with specialized prompts
   - Claude AI generates responses grounded in the provided context
   - Conversation history informs follow-up interactions

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- API keys for Claude AI and OpenAI/Qwen Embeddings

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/nexusrag.git
cd nexusrag
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:

```
ANTHROPIC_LLM_API_KEY=your_anthropic_api_key
OPENAI_EMBEDDING_API_KEY=your_openai_api_key
ANTHROPIC_MODEL_NAME=claude-3-7-sonnet
OPENAI_EMBEDDING_MODEL_NAME=Qwen3-Embedding-4B
```

4. **Run the application**

```bash
streamlit run app/web_chatbot.py
```

5. **Access the web interface**

Open your browser and go to `http://localhost:8501`

## 💡 Usage Guide

### Document Management

- **Upload Documents**: Click "Upload New Document" in the sidebar and select your files
- **View Knowledge Base**: See all uploaded documents in the sidebar
- **Remove Documents**: Delete individual documents or clear the entire knowledge base

### Asking Questions

- **Simple Queries**: "What are the key points in the executive summary?"
- **Comparative Analysis**: "Compare the financial results from 2023 to 2024"
- **Data Extraction**: "Extract all tables from section 3 and format them"
- **Summarization**: "Summarize the methodology section in bullet points"
- **Multi-document Questions**: "What do these documents say about climate change?"

## ⚙️ Advanced Configuration

Fine-tune NexusRAG by adjusting parameters in `src/config.py`:

```python
# LLM configuration
ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet"  # Model selection
TEMPERATURE = 0.7                           # Response creativity (0.0-1.0)
MAX_TOKENS = 2000                           # Maximum response length

# Retrieval parameters
TOP_K = 5                                   # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7                  # Minimum relevance score

# Chunking parameters
CHUNK_SIZE = 1000                           # Text chunk size
CHUNK_OVERLAP = 200                         # Overlap between chunks
```

## 🔮 Future Roadmap

- **Multi-language Support**: Expand capabilities to process and respond in multiple languages
- **Document Comparison**: Add specialized tools for side-by-side document analysis
- **Custom Knowledge Bases**: Support for creating and switching between multiple knowledge domains
- **Advanced Analytics**: Visualization of document relationships and knowledge graphs
- **API Integration**: REST API for programmatic access to the RAG capabilities

## 🛠️ Technology Stack

- **LangChain**: Framework for building LLM applications
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **Claude AI**: Advanced language model from Anthropic
- **Streamlit**: Interactive web interface
- **Qwen Embeddings**: High-quality text embeddings

## 📚 Best Practices

- **Document Quality**: Higher quality documents yield better results
- **Specific Questions**: More specific queries tend to get more precise answers
- **Context Awareness**: The system maintains conversation history, so follow-up questions work well
- **Source Verification**: Always check the provided source references for critical information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with 💙 for knowledge seekers everywhere
