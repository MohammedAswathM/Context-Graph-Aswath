import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.ontology.loader import OntologyLoader
from src.graph.neo4j_store import Neo4jStore
from src.processing.pipeline import ExtractionPipeline


def main():
    """Run sample extraction"""
    
    print("=" * 80)
    print("Context Graph Sample Usage")
    print("=" * 80)
    print()
    
    # 1. Initialize
    print("1. Initializing components...")
    settings = get_settings()
    
    loader = OntologyLoader()
    store = Neo4jStore(settings.neo4j)
    pipeline = ExtractionPipeline(loader, store)
    print("   ✓ Components initialized")
    print()
    
    # 2. Load ontology
    print("2. Loading business ontology...")
    ontology_path = Path(__file__).parent.parent / "ontologies" / "business_context.json"
    ontology = loader.load_from_file(ontology_path)
    print(f"   ✓ Loaded '{ontology.metadata.ontology_id}' ontology")
    print(f"     - {len(ontology.classes)} classes")
    print(f"     - {len(ontology.object_properties)} object properties")
    print(f"     - {len(ontology.datatype_properties)} datatype properties")
    print()
    
    # 3. Sample text
    print("3. Processing sample text...")
    
    sample_text = """
    TechVentures Inc. is a leading software development company founded in 2015.
    The organization is headquartered in Austin, Texas and employs approximately 250 people.
    
    Sarah Johnson is the CEO and founder of TechVentures. She leads the company's strategic 
    initiatives and oversees the product development division.
    
    The company's flagship product is CloudSync Pro, a cloud-based collaboration platform 
    priced at $29.99 per user per month. TechVentures also produces DataFlow Analytics, 
    an enterprise data visualization tool.
    
    Michael Chen serves as the CTO and manages the engineering team. He works closely with 
    Sarah on technical strategy and innovation.
    
    TechVentures has partnerships with several major corporations including GlobalTech Corp 
    and InnovateSystems. The company recently signed a contract with MegaCorp Industries 
    to provide enterprise software solutions.
    
    The company's website is www.techventures.com and they can be reached at 
    contact@techventures.com.
    """
    
    # 4. Extract knowledge
    print("   Processing text...")
    batches = pipeline.process_text(
        sample_text,
        ontology_id="business",
        collection="demo"
    )
    
    total_triples = sum(len(b.triples) for b in batches)
    print(f"   ✓ Extracted {total_triples} triples")
    print()
    
    # 5. Query results
    print("4. Querying extracted knowledge...")
    
    # List entities
    entities = store.get_all_entities(collection="demo", limit=10)
    print(f"\n   Found {len(entities)} entities:")
    for entity in entities[:5]:
        print(f"     - {entity}")
    
    # Query specific entity
    if entities:
        print(f"\n5. Querying entity '{entities[0]}'...")
        paths = store.query_entity(entities[0], collection="demo")
        print(f"   Found {len(paths)} relationship paths")
    
    print()
    
    # 6. Show sample queries
    print("6. Sample Neo4j Queries:")
    print()
    print("   Run these in Neo4j Browser (http://localhost:7474):")
    print()
    print("   // Find all organizations")
    print("   MATCH (n:Node {collection: 'demo'})")
    print("   RETURN n.uri")
    print()
    print("   // Find entity relationships")
    print("   MATCH (e:Node {collection: 'demo'})-[r:Rel]->(related)")
    print("   RETURN e.uri, r.uri, related")
    print("   LIMIT 50")
    print()
    
    # 7. Cleanup option
    print("7. Cleanup")
    print("   To delete this demo collection, run:")
    print("   >>> store.delete_collection('default', 'demo')")
    print()
    
    # Close
    store.close()
    
    print("=" * 80)
    print("✓ Sample completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Modify ontologies/business_context.json for your domain")
    print("  2. Use CLI: python -m src.cli process-file your_doc.txt --ontology-id business")
    print("  3. Query graph using Neo4j Browser at http://localhost:7474")
    print()


if __name__ == "__main__":
    main()