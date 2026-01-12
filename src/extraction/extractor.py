from typing import List
from ..ontology.models import Ontology
from ..graph.models import Triple, TriplesBatch, TripleMetadata
from .llm_client import LLMClient
from .prompts import PromptBuilder
from .parser import ResponseParser


class KnowledgeExtractor:
    """Extract knowledge from text using ontology-guided LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()
    
    def extract_from_text(
        self,
        text: str,
        ontology: Ontology,
        user: str = "default",
        collection: str = "default"
    ) -> TriplesBatch:
        """
        Extract triples from text using ontology.
        
        Args:
            text: Text to extract knowledge from
            ontology: Ontology defining entities and relationships
            user: User/tenant ID
            collection: Collection/namespace
        
        Returns:
            TriplesBatch with extracted triples
        """
        print(f"Extracting knowledge from text ({len(text)} chars)")
        
        # Build prompt
        prompt = self.prompt_builder.build_extraction_prompt(ontology, text)
        
        # Call LLM
        response = self.llm_client.extract(prompt)
        
        # Parse response
        result = self.parser.parse(response, ontology.metadata.ontology_id)
        
        if result is None:
            print("Failed to parse LLM response")
            return TriplesBatch(
                triples=[],
                metadata=TripleMetadata(user=user, collection=collection)
            )
        
        print(f"Extracted {len(result.triples)} triples")
        
        # Create batch
        batch = TriplesBatch(
            triples=result.triples,
            metadata=TripleMetadata(
                user=user,
                collection=collection,
                source=text[:100]  # First 100 chars as source reference
            )
        )
        
        return batch
    
    def extract_from_chunks(
        self,
        chunks: List[str],
        ontology: Ontology,
        user: str = "default",
        collection: str = "default"
    ) -> List[TriplesBatch]:
        """Extract from multiple text chunks"""
        print(f"Extracting from {len(chunks)} chunks")
        
        batches = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            batch = self.extract_from_text(chunk, ontology, user, collection)
            batches.append(batch)
        
        total_triples = sum(len(b.triples) for b in batches)
        print(f"Extracted {total_triples} total triples from {len(chunks)} chunks")
        
        return batches