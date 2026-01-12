# Context Graph - Ontology-Driven Knowledge Extraction

Build robust context graphs from unstructured text using ontology-constrained LLM extraction and Neo4j storage.

Inspired by TrustGraph architecture, optimized for standalone deployment.

## 🎯 Features

- **Ontology-Driven Extraction**: Define your domain model, get structured knowledge
- **LLM-Powered**: Uses GPT-4/Claude for intelligent entity and relationship extraction
- **Neo4j Storage**: Scalable property graph with multi-tenant support
- **Complete Pipeline**: Chunking → Extraction → Validation → Storage
- **CLI Interface**: Easy-to-use command-line tools

## 📋 Prerequisites

- Python 3.10 or higher
- Docker Desktop (for Neo4j)
- OpenAI or Anthropic API key

## 🚀 Quick Start

### Step 1: Clone/Setup Project

```bash
# If you haven't created the project structure yet:
mkdir context-graph && cd context-graph

# Create all directories
mkdir -p src/{ontology,extraction,graph,processing}
mkdir -p ontologies examples tests

# Create __init__.py files
touch src/__init__.py
touch src/{ontology,extraction,graph,processing}/__init__.py
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# - Add your OpenAI or Anthropic API key
# - Configure Neo4j credentials (default is fine for local)
```

Example `.env`:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=contextgraph123
NEO4J_DATABASE=contextgraph

# LLM Configuration (add your key)
OPENAI_API_KEY=sk-your-key-here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4-turbo-preview

# Application Settings
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Step 4: Start Neo4j

```bash
# Start Neo4j using Docker Compose
docker-compose up -d

# Wait for Neo4j to be ready (about 30 seconds)
docker-compose logs -f neo4j
# Wait for "Started." message, then Ctrl+C

# Verify Neo4j is running
docker-compose ps
```

Access Neo4j Browser at `http://localhost:7474`:
- Username: `neo4j`
- Password: `contextgraph123`

### Step 5: Run Sample

```bash
# Run the sample usage example
python examples/sample_usage.py
```

You should see:
```
Context Graph Sample Usage
==================================================
1. Initializing components...
   ✓ Components initialized

2. Loading business ontology...
   ✓ Loaded 'business' ontology
     - 10 classes
     - 15 object properties
     - 14 datatype properties

3. Processing sample text...
   ✓ Extracted 45 triples

4. Querying extracted knowledge...
   Found 8 entities
```

## 📚 Usage Guide

### Using the CLI

The CLI provides easy access to all functionality:

```bash
# Load an ontology
python -m src.cli load-ontology ontologies/business_context.json

# Extract from text
python -m src.cli extract "ACME Corp employs John Doe as CEO." \
  --ontology-id business \
  --collection demo

# Process a document
python -m src.cli process-file my_document.txt \
  --ontology-id business \
  --collection my_collection

# List extracted entities
python -m src.cli list-entities --collection demo --limit 20
```

### Using Python API

```python
from src.config import get_settings
from src.ontology.loader import OntologyLoader
from src.graph.neo4j_store import Neo4jStore
from src.processing.pipeline import ExtractionPipeline

# Initialize
settings = get_settings()
loader = OntologyLoader()
store = Neo4jStore(settings.neo4j)
pipeline = ExtractionPipeline(loader, store)

# Load ontology
loader.load_from_file("ontologies/business_context.json")

# Process text
text = "Your business document text here..."
batches = pipeline.process_text(
    text,
    ontology_id="business",
    collection="my_collection"
)

# Query results
entities = store.get_all_entities(collection="my_collection")
print(f"Extracted {len(entities)} entities")

# Close connection
store.close()
```

## 🎨 Creating Your Own Ontology

The ontology defines what entities and relationships the system can extract.

### Ontology Structure

```json
{
  "metadata": {
    "ontology_id": "your_domain",
    "version": "1.0.0",
    "description": "Your domain ontology"
  },
  "classes": {
    "YourEntity": {
      "uri": "your:YourEntity",
      "type": "owl:Class",
      "rdfs:label": [{"@value": "Your Entity", "@language": "en"}],
      "rdfs:comment": "Description of your entity type"
    }
  },
  "object_properties": {
    "yourRelationship": {
      "uri": "your:yourRelationship",
      "type": "owl:ObjectProperty",
      "rdfs:label": [{"@value": "your relationship", "@language": "en"}],
      "rdfs:domain": "SourceEntity",
      "rdfs:range": "TargetEntity",
      "rdfs:comment": "Describes the relationship"
    }
  },
  "datatype_properties": {
    "yourAttribute": {
      "uri": "your:yourAttribute",
      "type": "owl:DatatypeProperty",
      "rdfs:label": [{"@value": "your attribute", "@language": "en"}],
      "rdfs:domain": "YourEntity",
      "rdfs:range": "xsd:string",
      "rdfs:comment": "Describes the attribute"
    }
  }
}
```

### Best Practices

1. **Start Small**: Begin with 5-10 core entity types
2. **Be Specific**: Clear descriptions help the LLM extract accurately
3. **Use Constraints**: Domain/range constraints improve extraction quality
4. **Test Iteratively**: Extract from sample documents and refine

## 🔍 Querying Your Graph

### Neo4j Browser (Visual)

1. Open `http://localhost:7474`
2. Login with `neo4j` / `contextgraph123`
3. Run Cypher queries:

```cypher
// Find all organizations
MATCH (n:Node {collection: 'demo'})
RETURN n.uri
LIMIT 25

// Find relationships
MATCH (e:Node {collection: 'demo'})-[r:Rel]->(related)
RETURN e.uri, r.uri, related
LIMIT 50

// Find entities by type
MATCH (n:Node)-[:Rel {uri: 'rdf:type'}]->(t:Node)
WHERE n.collection = 'demo'
RETURN n.uri, t.uri
```

### Python API

```python
from src.graph.neo4j_store import Neo4jStore
from src.config import get_settings

store = Neo4jStore(get_settings().neo4j)

# Get all entities in a collection
entities = store.get_all_entities(collection="demo", limit=100)

# Query entity relationships
relationships = store.query_entity(
    entity_uri="business:techventures",
    collection="demo",
    max_depth=2
)

store.close()
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────┐
│              INPUT: Text Document                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│          Text Chunker (Overlap)                │
│  • Split into manageable pieces                │
│  • Preserve context with overlap               │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│         Ontology-Guided Extraction             │
│  • Load domain ontology                        │
│  • Build constrained prompt                    │
│  • LLM extracts structured triples             │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│          Response Parser                       │
│  • Parse JSON from LLM                         │
│  • Normalize URIs                              │
│  • Convert to RDF triples                      │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│          Neo4j Storage                         │
│  • (:Node) - Entity nodes                      │
│  • (:Literal) - Attribute values               │
│  • [:Rel] - Relationships                      │
│  • Multi-tenant collections                    │
└────────────────────────────────────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extraction.py

# Run with verbose output
pytest -v
```

## 🛠️ Troubleshooting

### Neo4j Connection Error

```bash
# Check if Neo4j is running
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### LLM Extraction Fails

1. Check API key in `.env`
2. Verify model name is correct
3. Check API rate limits
4. Try simpler text first

### No Triples Extracted

1. Ensure ontology is loaded
2. Verify text matches ontology domain
3. Check LLM response in logs (set `LOG_LEVEL=DEBUG`)
4. Test with sample text from `examples/sample_usage.py`

### Import Errors

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## 📈 Performance Tips

### For Large Documents

```python
# Adjust chunk size in .env
CHUNK_SIZE=2000
CHUNK_OVERLAP=400

# Or in code:
from src.processing.chunker import TextChunker

chunker = TextChunker(chunk_size=2000, chunk_overlap=400)
```

### For High Volume

```python
# Batch processing
from src.graph.models import Triple, TriplesBatch, TripleMetadata

# Collect many triples
triples = []  # ... your triples

# Store in single batch
batch = TriplesBatch(
    triples=triples,
    metadata=TripleMetadata(user="default", collection="demo")
)
store.add_triples_batch(batch)
```

### Neo4j Optimization

```bash
# In docker-compose.yml, increase memory:
NEO4J_dbms_memory_heap_initial__size: 4G
NEO4J_dbms_memory_heap_max__size: 8G
```

## 🎓 Next Steps

1. **Customize Your Ontology**: Edit `ontologies/business_context.json` for your domain
2. **Process Your Documents**: Use CLI to extract from your text files
3. **Explore the Graph**: Query in Neo4j Browser to find insights
4. **Iterate and Improve**: Refine ontology based on extraction results
5. **Scale Up**: Add more documents and collections

## 📖 Learn More

- **TrustGraph**: https://github.com/trustgraph-ai/trustgraph
- **Neo4j Docs**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/developer/cypher/

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 💡 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the sample code in `examples/`
3. Open a GitHub issue with details

---

**Built with inspiration from TrustGraph's robust architecture**

Happy graph building! 🚀