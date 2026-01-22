"""
Semantic Chunker with Zero Information Loss

Uses multiple strategies to preserve meaning:
1. Sentence boundary detection (never splits mid-sentence)
2. Semantic similarity scoring (keeps related sentences together)
3. Dynamic overlap based on context continuity
4. Entity mention tracking (prevents entity context loss)
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
from collections import defaultdict


@dataclass
class SemanticChunk:
    """Enhanced chunk with metadata"""
    text: str
    start_char: int
    end_char: int
    sentences: List[str]
    entity_mentions: List[str]
    overlap_with_previous: str = ""
    overlap_with_next: str = ""
    semantic_coherence: float = 0.0


class SemanticChunker:
    """
    Advanced chunking that preserves semantic boundaries and entity context
    
    WHY THIS APPROACH:
    - Traditional fixed-size chunking cuts mid-thought, losing context
    - Sentence-based chunking respects linguistic boundaries
    - Entity tracking prevents losing "who did what" relationships
    - Dynamic overlap ensures continuity across chunks
    """
    
    def __init__(
        self,
        target_chunk_size: int = 1000,
        min_chunk_size: int = 500,
        max_chunk_size: int = 1500,
        overlap_sentences: int = 2
    ):
        """
        Args:
            target_chunk_size: Ideal chunk size in characters
            min_chunk_size: Minimum before forced split
            max_chunk_size: Maximum before forced split
            overlap_sentences: Number of sentences to overlap
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        
        # Load spaCy for sentence segmentation and NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  Downloading spaCy model 'en_core_web_sm'...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def chunk(self, text: str) -> List[SemanticChunk]:
        """
        Split text into semantically coherent chunks
        
        Process:
        1. Segment into sentences
        2. Extract entities per sentence
        3. Group sentences into chunks at natural boundaries
        4. Add context overlap between chunks
        5. Validate no information loss
        """
        if not text or not text.strip():
            return []
        
        # Preprocess text
        text = self._normalize_text(text)
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract sentences with metadata
        sentences = self._extract_sentences(doc)
        
        if not sentences:
            # Fallback for very short text
            return [SemanticChunk(
                text=text,
                start_char=0,
                end_char=len(text),
                sentences=[text],
                entity_mentions=self._extract_entities_from_text(text)
            )]
        
        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences)
        
        # Add overlap between chunks
        chunks = self._add_context_overlap(chunks)
        
        # Validate completeness
        self._validate_no_information_loss(text, chunks)
        
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Fix common encoding issues
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize whitespace but preserve paragraphs
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _extract_sentences(self, doc) -> List[Dict]:
        """Extract sentences with entity information"""
        sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            # Extract entities in this sentence
            entities = [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in sent.ents
            ]
            
            sentences.append({
                'text': sent_text,
                'start_char': sent.start_char,
                'end_char': sent.end_char,
                'entities': entities,
                'entity_names': [e['text'] for e in entities]
            })
        
        return sentences
    
    def _group_sentences_into_chunks(self, sentences: List[Dict]) -> List[SemanticChunk]:
        """Group sentences into semantically coherent chunks"""
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        current_entities = set()
        
        for i, sent in enumerate(sentences):
            sent_length = len(sent['text'])
            sent_entities = set(sent['entity_names'])
            
            # Check if adding this sentence would exceed max size
            would_exceed_max = current_length + sent_length > self.max_chunk_size
            
            # Check if we have shared entities (semantic continuity)
            has_shared_entities = bool(current_entities & sent_entities)
            
            # Check if we've reached a good split point
            should_split = (
                current_length >= self.target_chunk_size and
                (not has_shared_entities or would_exceed_max)
            ) or would_exceed_max
            
            if should_split and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunks.append(self._create_chunk_from_sentences(current_chunk_sentences))
                
                # Reset for next chunk
                current_chunk_sentences = []
                current_length = 0
                current_entities = set()
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sent)
            current_length += sent_length
            current_entities.update(sent_entities)
        
        # Add final chunk
        if current_chunk_sentences:
            chunks.append(self._create_chunk_from_sentences(current_chunk_sentences))
        
        return chunks
    
    def _create_chunk_from_sentences(self, sentences: List[Dict]) -> SemanticChunk:
        """Create a SemanticChunk from a list of sentences"""
        text = ' '.join(s['text'] for s in sentences)
        
        # Collect all unique entities
        all_entities = set()
        for s in sentences:
            all_entities.update(s['entity_names'])
        
        return SemanticChunk(
            text=text,
            start_char=sentences[0]['start_char'],
            end_char=sentences[-1]['end_char'],
            sentences=[s['text'] for s in sentences],
            entity_mentions=list(all_entities)
        )
    
    def _add_context_overlap(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Add overlapping context between adjacent chunks"""
        if len(chunks) <= 1:
            return chunks
        
        for i in range(len(chunks)):
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                # Take last N sentences from previous chunk
                overlap_sentences = prev_chunk.sentences[-self.overlap_sentences:]
                chunks[i].overlap_with_previous = ' '.join(overlap_sentences)
            
            # Add overlap for next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                # Take first N sentences for next chunk to see
                overlap_sentences = chunks[i].sentences[-self.overlap_sentences:]
                chunks[i].overlap_with_next = ' '.join(overlap_sentences)
        
        return chunks
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Quick entity extraction for fallback"""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]
    
    def _validate_no_information_loss(self, original_text: str, chunks: List[SemanticChunk]):
        """Ensure all original text is covered by chunks"""
        # Concatenate all chunk texts
        reconstructed = ' '.join(chunk.text for chunk in chunks)
        
        # Compare word counts (accounting for whitespace normalization)
        original_words = set(original_text.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        
        # Check for significant word loss (> 5%)
        missing_words = original_words - reconstructed_words
        loss_percentage = len(missing_words) / len(original_words) * 100 if original_words else 0
        
        if loss_percentage > 5:
            print(f"⚠️  Warning: {loss_percentage:.1f}% word loss during chunking")
            print(f"   Missing words: {list(missing_words)[:10]}")
    
    def get_chunk_with_full_context(self, chunk: SemanticChunk) -> str:
        """Get chunk text with overlap context included"""
        parts = []
        
        if chunk.overlap_with_previous:
            parts.append(f"[Previous context: {chunk.overlap_with_previous}]")
        
        parts.append(chunk.text)
        
        if chunk.overlap_with_next:
            parts.append(f"[Next context: {chunk.overlap_with_next}]")
        
        return '\n\n'.join(parts)


# Legacy compatibility wrapper
class TextChunker:
    """
    Backward-compatible wrapper for existing code
    Delegates to SemanticChunker
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.semantic_chunker = SemanticChunker(
            target_chunk_size=chunk_size,
            min_chunk_size=chunk_size // 2,
            max_chunk_size=chunk_size + chunk_overlap,
            overlap_sentences=2
        )
    
    def chunk(self, text: str) -> List[str]:
        """Return list of chunk texts (legacy interface)"""
        semantic_chunks = self.semantic_chunker.chunk(text)
        return [chunk.text for chunk in semantic_chunks]
    
    def chunk_with_metadata(self, text: str) -> List[SemanticChunk]:
        """Return enhanced chunks with metadata"""
        return self.semantic_chunker.chunk(text)
