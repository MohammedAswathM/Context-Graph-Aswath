from typing import List


class TextChunker:
    """Chunk text for processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text with overlap.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                last_period = chunk.rfind('.')
                last_exclaim = chunk.rfind('!')
                last_question = chunk.rfind('?')
                
                break_point = max(last_period, last_exclaim, last_question)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        print(f"Chunked text into {len(chunks)} chunks")
        return chunks