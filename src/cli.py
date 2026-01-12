import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .ontology.loader import OntologyLoader
from .graph.neo4j_store import Neo4jStore
from .processing.pipeline import ExtractionPipeline

console = Console()


def get_ontology_loader():
    """
    Get OntologyLoader and auto-load all ontologies from ontologies/ directory
    """
    loader = OntologyLoader()
    
    # Find ontologies directory
    ontologies_dir = Path("ontologies")
    
    if not ontologies_dir.exists():
        console.print("⚠️  Warning: ontologies/ directory not found", style="yellow")
        return loader
    
    # Load all JSON files from ontologies directory
    json_files = list(ontologies_dir.glob("*.json"))
    
    if not json_files:
        console.print("⚠️  Warning: No ontology files found in ontologies/", style="yellow")
        return loader
    
    # Load each ontology
    for json_file in json_files:
        try:
            loader.load_from_file(json_file)
            console.print(f"✓ Loaded ontology from {json_file.name}", style="dim")
        except Exception as e:
            console.print(f"⚠️  Failed to load {json_file.name}: {e}", style="yellow")
    
    return loader


@click.group()
def cli():
    """Context Graph CLI - Ontology-driven knowledge extraction"""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def load_ontology(file_path: str):
    """Load and validate an ontology from JSON file"""
    try:
        loader = OntologyLoader()
        ontology = loader.load_from_file(file_path)
        
        console.print(f"✓ Loaded ontology: {ontology.metadata.ontology_id}", style="green")
        console.print(f"  Classes: {len(ontology.classes)}")
        console.print(f"  Object Properties: {len(ontology.object_properties)}")
        console.print(f"  Datatype Properties: {len(ontology.datatype_properties)}")
        
        # Validate
        issues = ontology.validate_structure()
        if issues:
            console.print("\n⚠️  Validation issues:", style="yellow")
            for issue in issues:
                console.print(f"  - {issue}", style="yellow")
        else:
            console.print("\n✓ Ontology validation passed", style="green")
        
    except Exception as e:
        console.print(f"✗ Failed to load ontology: {e}", style="red")


@cli.command()
@click.option('--list-loaded', is_flag=True, help='List all loaded ontologies')
def list_ontologies(list_loaded: bool):
    """List available ontologies"""
    try:
        if list_loaded:
            # Load and show what's available
            loader = get_ontology_loader()
            ontologies = loader.get_all_ontologies()
            
            if not ontologies:
                console.print("No ontologies loaded", style="yellow")
                return
            
            table = Table(title="Loaded Ontologies")
            table.add_column("ID", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Classes", justify="right")
            table.add_column("Properties", justify="right")
            
            for ont_id, ont in ontologies.items():
                total_props = len(ont.object_properties) + len(ont.datatype_properties)
                table.add_row(
                    ont_id,
                    ont.metadata.version,
                    str(len(ont.classes)),
                    str(total_props)
                )
            
            console.print(table)
        else:
            # Just list files
            ontologies_dir = Path("ontologies")
            if not ontologies_dir.exists():
                console.print("ontologies/ directory not found", style="yellow")
                return
            
            json_files = list(ontologies_dir.glob("*.json"))
            
            if not json_files:
                console.print("No ontology files found", style="yellow")
                return
            
            console.print(f"\nFound {len(json_files)} ontology file(s):")
            for f in json_files:
                console.print(f"  • {f.name}", style="cyan")
        
    except Exception as e:
        console.print(f"✗ Failed: {e}", style="red")


@cli.command()
@click.argument('text')
@click.option('--ontology-id', required=True, help='Ontology to use')
@click.option('--collection', default='default', help='Collection name')
def extract(text: str, ontology_id: str, collection: str):
    """Extract knowledge from text"""
    try:
        settings = get_settings()
        
        # Initialize components with auto-loaded ontologies
        console.print("Initializing...", style="dim")
        loader = get_ontology_loader()
        store = Neo4jStore(settings.neo4j)
        pipeline = ExtractionPipeline(loader, store)
        
        # Verify ontology exists
        if not loader.get_ontology(ontology_id):
            console.print(f"\n✗ Ontology '{ontology_id}' not found", style="red")
            console.print(f"\nAvailable ontologies:", style="yellow")
            for ont_id in loader.get_all_ontologies().keys():
                console.print(f"  • {ont_id}", style="cyan")
            store.close()
            return
        
        console.print(f"Extracting knowledge using ontology '{ontology_id}'...")
        batches = pipeline.process_text(
            text, ontology_id, collection=collection
        )
        
        # Show results
        total_triples = sum(len(b.triples) for b in batches)
        console.print(f"✓ Extracted {total_triples} triples", style="green")
        
        # Show sample entities
        entities = store.get_all_entities(collection=collection, limit=5)
        if entities:
            console.print(f"\nExtracted entities:", style="dim")
            for entity in entities[:5]:
                console.print(f"  • {entity}", style="cyan")
        
        store.close()
        
    except Exception as e:
        console.print(f"✗ Extraction failed: {e}", style="red")
        import traceback
        if settings.log_level == "DEBUG":
            traceback.print_exc()


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--ontology-id', required=True, help='Ontology to use')
@click.option('--collection', default='default', help='Collection name')
def process_file(file_path: str, ontology_id: str, collection: str):
    """Process document file"""
    try:
        settings = get_settings()
        
        # Initialize components with auto-loaded ontologies
        console.print("Initializing...", style="dim")
        loader = get_ontology_loader()
        store = Neo4jStore(settings.neo4j)
        pipeline = ExtractionPipeline(loader, store)
        
        # Verify ontology exists
        if not loader.get_ontology(ontology_id):
            console.print(f"\n✗ Ontology '{ontology_id}' not found", style="red")
            console.print(f"\nAvailable ontologies:", style="yellow")
            for ont_id in loader.get_all_ontologies().keys():
                console.print(f"  • {ont_id}", style="cyan")
            store.close()
            return
        
        console.print(f"Processing file: {file_path}")
        batches = pipeline.process_document(
            file_path, ontology_id, collection=collection
        )
        
        total_triples = sum(len(b.triples) for b in batches)
        console.print(f"✓ Extracted {total_triples} triples", style="green")
        
        # Show sample entities
        entities = store.get_all_entities(collection=collection, limit=5)
        if entities:
            console.print(f"\nSample extracted entities:", style="dim")
            for entity in entities[:5]:
                console.print(f"  • {entity}", style="cyan")
        
        store.close()
        
    except Exception as e:
        console.print(f"✗ Processing failed: {e}", style="red")
        import traceback
        settings = get_settings()
        if settings.log_level == "DEBUG":
            traceback.print_exc()


@cli.command()
@click.option('--collection', default='default', help='Collection name')
@click.option('--limit', default=20, help='Max entities to show')
def list_entities(collection: str, limit: int):
    """List entities in collection"""
    try:
        settings = get_settings()
        store = Neo4jStore(settings.neo4j)
        
        entities = store.get_all_entities(collection=collection, limit=limit)
        
        if not entities:
            console.print(f"No entities found in collection '{collection}'", style="yellow")
            store.close()
            return
        
        table = Table(title=f"Entities in '{collection}'")
        table.add_column("Entity URI", style="cyan")
        
        for entity in entities:
            table.add_row(entity)
        
        console.print(table)
        console.print(f"\nShowing {len(entities)} of max {limit} entities", style="dim")
        
        store.close()
        
    except Exception as e:
        console.print(f"✗ Failed: {e}", style="red")


@cli.command()
@click.argument('entity_uri')
@click.option('--collection', default='default', help='Collection name')
@click.option('--depth', default=2, help='Max relationship depth')
def query_entity(entity_uri: str, collection: str, depth: int):
    """Query entity relationships"""
    try:
        settings = get_settings()
        store = Neo4jStore(settings.neo4j)
        
        console.print(f"Querying entity: {entity_uri}")
        results = store.query_entity(entity_uri, collection=collection, max_depth=depth)
        
        console.print(f"\n✓ Found {len(results)} relationship paths", style="green")
        
        if results and len(results) > 0:
            console.print("\nUse Neo4j Browser for visualization: http://localhost:7474")
            console.print(f"\nSample Cypher query:")
            console.print(f"  MATCH (e:Node {{uri: '{entity_uri}', collection: '{collection}'}})-[r:Rel]-(related)")
            console.print(f"  RETURN e, r, related LIMIT 50")
        
        store.close()
        
    except Exception as e:
        console.print(f"✗ Query failed: {e}", style="red")


@cli.command()
@click.option('--collection', required=True, help='Collection name to delete')
@click.option('--user', default='default', help='User/tenant')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_collection(collection: str, user: str, confirm: bool):
    """Delete a collection and all its data"""
    try:
        if not confirm:
            response = input(f"⚠️  Delete collection '{user}/{collection}' and ALL its data? (yes/no): ")
            if response.lower() != 'yes':
                console.print("Cancelled", style="yellow")
                return
        
        settings = get_settings()
        store = Neo4jStore(settings.neo4j)
        
        store.delete_collection(user, collection)
        console.print(f"✓ Deleted collection '{user}/{collection}'", style="green")
        
        store.close()
        
    except Exception as e:
        console.print(f"✗ Failed: {e}", style="red")


@cli.command()
def info():
    """Show system information"""
    try:
        settings = get_settings()
        
        console.print("\n📊 Context Graph System Info\n", style="bold")
        
        # LLM settings
        console.print("🤖 LLM Configuration:", style="bold cyan")
        console.print(f"  Provider: {settings.llm.provider}")
        console.print(f"  Model: {settings.llm.model}")
        console.print(f"  Temperature: {settings.llm.temperature}")
        
        # Neo4j settings
        console.print("\n🗄️  Neo4j Configuration:", style="bold cyan")
        console.print(f"  URI: {settings.neo4j.uri}")
        console.print(f"  Database: {settings.neo4j.database}")
        
        # Processing settings
        console.print("\n⚙️  Processing Configuration:", style="bold cyan")
        console.print(f"  Chunk size: {settings.processing.chunk_size}")
        console.print(f"  Chunk overlap: {settings.processing.chunk_overlap}")
        
        # Ontologies
        console.print("\n📚 Available Ontologies:", style="bold cyan")
        ontologies_dir = Path("ontologies")
        if ontologies_dir.exists():
            json_files = list(ontologies_dir.glob("*.json"))
            if json_files:
                for f in json_files:
                    console.print(f"  • {f.name}")
            else:
                console.print("  (none found)")
        else:
            console.print("  (ontologies/ directory not found)")
        
        console.print()
        
    except Exception as e:
        console.print(f"✗ Failed: {e}", style="red")


if __name__ == '__main__':
    cli()