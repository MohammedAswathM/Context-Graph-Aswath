from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import re


class SearchStrategy(Enum):
    """Search strategy types"""
    VECTOR = "vector"  # Semantic/similarity search
    GRAPH = "graph"    # Relationship traversal
    HYBRID = "hybrid"  # Combined approach


class QueryIntent(BaseModel):
    """Parsed query intent"""
    query_text: str
    strategy: SearchStrategy
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    extracted_entities: List[str] = Field(default_factory=list)
    relationship_type: Optional[str] = None


class SearchRouter:
    """
    Intelligent query router using pattern matching and heuristics.
    Routes queries to optimal search strategy.
    """
    
    # Semantic similarity keywords (vector search)
    SIMILARITY_KEYWORDS = [
        'similar', 'like', 'related', 'comparable', 'alike',
        'resembling', 'matching', 'corresponding', 'parallel',
        'analogous', 'equivalent', 'akin'
    ]
    
    # Relationship keywords (graph search)
    RELATIONSHIP_KEYWORDS = [
        'works for', 'employed by', 'manages', 'reports to',
        'owns', 'operates', 'leads', 'founded', 'created',
        'part of', 'belongs to', 'member of', 'connected to',
        'relationship', 'link', 'connection'
    ]
    
    # Question words that don't indicate entities
    QUESTION_WORDS = [
        'what', 'who', 'where', 'when', 'why', 'how',
        'which', 'whose', 'whom', 'is', 'are', 'was', 'were'
    ]
    
    def __init__(self):
        """Initialize search router"""
        pass
    
    def route_query(self, query: str) -> QueryIntent:
        """
        Route query to optimal search strategy.
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent with strategy and metadata
        """
        query_lower = query.lower().strip()
        
        # Extract entities (excluding question words)
        entities = self._extract_entities(query)
        
        # Score each strategy
        vector_score = self._score_vector_search(query_lower, entities)
        graph_score = self._score_graph_search(query_lower, entities)
        
        # Determine strategy
        if vector_score > graph_score and vector_score > 0.5:
            strategy = SearchStrategy.VECTOR
            confidence = vector_score
            reasoning = f"High similarity keyword score: {vector_score:.2f} | Entities: {', '.join(entities) if entities else 'none'}"
        elif graph_score > vector_score and graph_score > 0.5:
            strategy = SearchStrategy.GRAPH
            confidence = graph_score
            rel_type = self._extract_relationship_type(query_lower)
            reasoning = f"High relationship keyword score: {graph_score:.2f} | Relationship: {rel_type or 'implicit'}"
        elif vector_score > 0.3 and graph_score > 0.3:
            # Both scores moderate - use hybrid
            strategy = SearchStrategy.HYBRID
            confidence = (vector_score + graph_score) / 2
            reasoning = f"Mixed signals (vector: {vector_score:.2f}, graph: {graph_score:.2f})"
        else:
            # Default to hybrid for ambiguous queries
            strategy = SearchStrategy.HYBRID
            confidence = max(vector_score, graph_score, 0.3)
            reasoning = f"Ambiguous query - using hybrid | Found entities: {', '.join(entities) if entities else 'none'}"
        
        return QueryIntent(
            query_text=query,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            extracted_entities=entities,
            relationship_type=self._extract_relationship_type(query_lower) if strategy == SearchStrategy.GRAPH else None
        )
    
    def _score_vector_search(self, query: str, entities: List[str]) -> float:
        """
        Score query for vector search suitability.
        
        Indicators:
        - Similarity keywords ("like", "similar")
        - Conceptual questions ("find companies...")
        - Multiple entities for comparison
        """
        score = 0.0
        
        # Check for similarity keywords
        for keyword in self.SIMILARITY_KEYWORDS:
            if keyword in query:
                score += 0.4
                break
        
        # Boost for conceptual terms
        conceptual_terms = ['concept', 'theme', 'nature', 'type', 'category', 'kind']
        if any(term in query for term in conceptual_terms):
            score += 0.2
        
        # Boost for "find" + entities (exploratory search)
        if 'find' in query and len(entities) > 0:
            score += 0.2
        
        # Boost for comparison patterns
        comparison_patterns = [
            r'like\s+\w+',      # "like Apple"
            r'similar to\s+\w+', # "similar to Google"
            r'as\s+\w+\s+as',   # "as big as Microsoft"
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, query):
                score += 0.3
                break
        
        return min(score, 1.0)
    
    def _score_graph_search(self, query: str, entities: List[str]) -> float:
        """
        Score query for graph traversal suitability.
        
        Indicators:
        - Relationship keywords ("works for", "manages")
        - Explicit entity mentions
        - Question words asking about connections ("who works at...")
        """
        score = 0.0
        
        # Check for relationship keywords
        for keyword in self.RELATIONSHIP_KEYWORDS:
            if keyword in query:
                score += 0.5
                break
        
        # Boost for entity presence
        if len(entities) > 0:
            score += 0.2
            # Extra boost for multiple entities (potential relationship query)
            if len(entities) >= 2:
                score += 0.1
        
        # Boost for "who/what + verb" patterns (relationship questions)
        relationship_patterns = [
            r'who\s+(works|manages|leads|owns)',
            r'what\s+(companies|organizations)\s+(employ|own|operate)',
            r'how many\s+\w+\s+(work at|belong to)',
        ]
        for pattern in relationship_patterns:
            if re.search(pattern, query):
                score += 0.4
                break
        
        # Penalize similarity language
        if any(keyword in query for keyword in self.SIMILARITY_KEYWORDS):
            score -= 0.3
        
        return max(0.0, min(score, 1.0))
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract potential entity mentions from query.
        
        Uses:
        - Capitalized words (proper nouns)
        - Known entity patterns
        - Excludes question words
        """
        entities = []
        
        # Remove question words at start
        words = query.split()
        filtered_words = [
            w for w in words 
            if w.lower() not in self.QUESTION_WORDS
        ]
        
        # Find capitalized words (potential proper nouns)
        for word in filtered_words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Not a question word
                if clean_word.lower() not in self.QUESTION_WORDS:
                    entities.append(clean_word)
        
        return entities
    
    def _extract_relationship_type(self, query: str) -> Optional[str]:
        """Extract relationship type from query"""
        # Map keywords to relationship types
        relationship_map = {
            'works for': 'employs',
            'employed by': 'employs',
            'manages': 'manages',
            'reports to': 'reportsTo',
            'owns': 'owns',
            'founded': 'founded',
            'created': 'created',
        }
        
        for keyword, rel_type in relationship_map.items():
            if keyword in query:
                return rel_type
        
        return None


class AdaptiveSearchExecutor:
    """
    Executes searches using routed strategy.
    Tracks performance for continuous optimization.
    """
    
    def __init__(self, vector_store, graph_store):
        """
        Initialize executor.
        
        Args:
            vector_store: VectorSearchStore instance
            graph_store: Neo4jStore instance
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Performance tracking
        self.stats = {
            'vector': 0,
            'graph': 0,
            'hybrid': 0,
            'total_queries': 0
        }
    
    def search(
        self,
        query: str,
        collection: str = "default",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Execute routed search.
        
        Args:
            query: Search query
            collection: Collection to search
            limit: Max results
            
        Returns:
            Search results with metadata
        """
        router = SearchRouter()
        intent = router.route_query(query)
        
        # Track statistics
        self.stats['total_queries'] += 1
        self.stats[intent.strategy.value] += 1
        
        # Execute based on strategy
        if intent.strategy == SearchStrategy.VECTOR:
            results = self._execute_vector_search(query, collection, limit)
        elif intent.strategy == SearchStrategy.GRAPH:
            results = self._execute_graph_search(intent, collection, limit)
        else:  # HYBRID
            results = self._execute_hybrid_search(query, intent, collection, limit)
        
        return {
            'query': query,
            'strategy': intent.strategy.value,
            'confidence': intent.confidence,
            'reasoning': intent.reasoning,
            'results': results,
            'entities': intent.extracted_entities
        }
    
    def _execute_vector_search(self, query: str, collection: str, limit: int) -> List[Dict[str, Any]]:
        """Execute vector similarity search"""
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.search(query, collection=collection, limit=limit)
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []
    
    def _execute_graph_search(self, intent: QueryIntent, collection: str, limit: int) -> List[Dict[str, Any]]:
        """Execute graph traversal search"""
        if self.graph_store is None or not intent.extracted_entities:
            return []
        
        results = []
        for entity in intent.extracted_entities[:3]:  # Limit to 3 entities
            try:
                entity_results = self.graph_store.query_entity(
                    entity_uri=entity,
                    collection=collection,
                    max_depth=2
                )
                results.extend(entity_results[:limit])
            except Exception as e:
                print(f"Graph search failed for {entity}: {e}")
        
        return results[:limit]
    
    def _execute_hybrid_search(
        self,
        query: str,
        intent: QueryIntent,
        collection: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search (vector + graph)"""
        vector_results = self._execute_vector_search(query, collection, limit // 2)
        graph_results = self._execute_graph_search(intent, collection, limit // 2)
        
        # Combine and deduplicate
        combined = vector_results + graph_results
        return combined[:limit]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get search strategy distribution"""
        if self.stats['total_queries'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'distribution': {
                'vector': self.stats['vector'] / self.stats['total_queries'],
                'graph': self.stats['graph'] / self.stats['total_queries'],
                'hybrid': self.stats['hybrid'] / self.stats['total_queries']
            }
        }