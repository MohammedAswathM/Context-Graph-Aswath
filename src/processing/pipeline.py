from typing import List, Optional
from ..config import get_settings
from ..ontology.loader import OntologyLoader
from ..extraction.extractor import KnowledgeExtractor
from ..extraction.llm_client import LLMClient
from ..graph.neo4j_store import Neo4jStore
from ..graph.models import TriplesBatch
from .chunker import TextChunker


class ExtractionPipeline:
    """Complete extraction pipeline"""
    
    def __init__(
        self,
        ontology_loader: OntologyLoader,
        graph_store: Neo4jStore,
        llm_client: Optional[LLMClient] = None
    ):
        settings = get_settings()
        
        self.ontology_loader = ontology_loader
        self.graph_store = graph_store
        
        # Initialize components
        if llm_client is None:
            llm_client = LLMClient(settings.llm)
        
        self.extractor = KnowledgeExtractor(llm_client)
        self.chunker = TextChunker(
            chunk_size=settings.processing.chunk_size,
            chunk_overlap=settings.processing.chunk_overlap
        )
    
    def process_text(
        self,
        text: str,
        ontology_id: str,
        user: str = "default",
        collection: str = "default",
        store_triples: bool = True
    ) -> List[TriplesBatch]:
        """
        Complete processing pipeline.
        
        Args:
            text: Text to process
            ontology_id: ID of ontology to use
            user: User/tenant ID
            collection: Collection/namespace
            store_triples: Whether to store in graph database
        
        Returns:
            List of extracted triple batches
        """
        print(f"Processing text for {user}/{collection} with ontology {ontology_id}")
        
        # Get ontology
        ontology = self.ontology_loader.get_ontology(ontology_id)
        if ontology is None:
            raise ValueError(f"Ontology not found: {ontology_id}")
        
        # Chunk text
        chunks = self.chunker.chunk(text)
        print(f"Created {len(chunks)} chunks")
        
        # Extract triples
        batches = self.extractor.extract_from_chunks(
            chunks, ontology, user, collection
        )
        
        # Store in graph
        if store_triples:
            for batch in batches:
                if len(batch.triples) > 0:
                    self.graph_store.add_triples_batch(batch)
        
        total_triples = sum(len(b.triples) for b in batches)
        print(f"Pipeline complete: {total_triples} triples extracted")
        
        return batches
    
    def process_document(
        self,
        file_path: str,
        ontology_id: str,
        user: str = "default",
        collection: str = "default"
    ) -> List[TriplesBatch]:
        """Process document file"""
        print(f"Processing document: {file_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.process_text(text, ontology_id, user, collection)