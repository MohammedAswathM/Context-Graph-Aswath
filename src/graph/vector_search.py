"""
Vector Search with Neo4j Native HNSW Indexes

Implements semantic search over knowledge graph using:
1. Sentence Transformers for embedding generation
2. Neo4j native vector indexes (HNSW algorithm)
3. Hybrid search combining vector + graph traversal

WHY NEO4J NATIVE VECTORS over FAISS/separate vector DB:
- Unified storage: Vectors + graph in one database
- No synchronization issues between systems
- Native HNSW implementation (fast approximate nearest neighbor)
- Can combine similarity search with graph queries
- Scales to billions of vectors
- Transactional consistency

EMBEDDING STRATEGY:
- Entity embeddings: From entity name + context + type
- Relationship embeddings: From subject + predicate + object
- Aggregate embeddings: Mean/weighted for multi-hop paths
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import json


@dataclass
class SearchResult:
    """Result from vector search"""
    uri: str
    score: float  # Similarity score (0-1)
    entity_type: Optional[str] = None
    properties: Dict[str, Any] = None
    context: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class VectorIndexConfig:
    """Configuration for vector index"""
    dimension: int = 384  # all-MiniLM-L6-v2 dimension
    similarity_function: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "hnsw"  # Neo4j uses HNSW
    m: int = 16  # HNSW M parameter (connections per layer)
    ef_construction: int = 200  # HNSW construction time param
    ef_search: int = 100  # HNSW search time param


class EmbeddingGenerator:
    """
    Generate embeddings using Sentence Transformers
    
    WHY SENTENCE-TRANSFORMERS:
    - Pre-trained on semantic similarity
    - Efficient (384d vectors vs 1536d OpenAI)
    - Runs locally (no API costs)
    - Good for short texts (entity names, descriptions)
    
    MODEL CHOICE: all-MiniLM-L6-v2
    - 384 dimensions
    - Fast inference
    - Good quality/speed tradeoff
    - Perfect for entity/relationship embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: SentenceTransformer model name
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dimension}")
    
    def embed_entity(
        self,
        uri: str,
        name: str,
        entity_type: Optional[str] = None,
        context: Optional[str] = None
    ) -> np.ndarray:
        """
        Create entity embedding
        
        Combines:
        - Entity name (primary signal)
        - Entity type (semantic category)
        - Context (surrounding text)
        
        Returns:
            384-dimensional vector
        """
        # Build rich representation
        parts = [name]
        
        if entity_type:
            parts.append(f"[{entity_type}]")
        
        if context:
            # Add first 200 chars of context
            parts.append(context[:200])
        
        text = " ".join(parts)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_relationship(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        context: Optional[str] = None
    ) -> np.ndarray:
        """
        Create relationship embedding
        
        Combines triple into meaningful representation:
        "Subject predicate Object"
        
        Example: "TechVentures employs Sarah Johnson"
        """
        # Build triple text
        triple_text = f"{subject_name} {predicate} {object_name}"
        
        if context:
            triple_text += f" [{context[:100]}]"
        
        embedding = self.model.encode(triple_text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_text(self, text: str) -> np.ndarray:
        """Simple text embedding"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding / np.linalg.norm(embedding)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple texts (efficient)"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize each vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


class VectorSearchStore:
    """
    Extends Neo4j with vector search capabilities
    
    IMPLEMENTATION:
    - Creates vector indexes on Node entities
    - Stores embeddings as node properties
    - Provides similarity search via Neo4j queries
    - Supports hybrid search (vector + graph)
    """
    
    def __init__(
        self,
        neo4j_store,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        index_config: Optional[VectorIndexConfig] = None
    ):
        """
        Args:
            neo4j_store: Existing Neo4jStore instance
            embedding_generator: Embedding model
            index_config: Vector index configuration
        """
        self.neo4j_store = neo4j_store
        self.driver = neo4j_store.driver
        self.database = neo4j_store.settings.database
        
        # Initialize embedding generator
        self.embedder = embedding_generator or EmbeddingGenerator()
        
        # Index configuration
        self.config = index_config or VectorIndexConfig(
            dimension=self.embedder.dimension
        )
        
        # Create vector indexes
        self._create_vector_indexes()
    
    def _create_vector_indexes(self):
        """
        Create Neo4j vector indexes
        
        Neo4j 5.x supports native vector indexes using HNSW
        Syntax: CREATE VECTOR INDEX index_name FOR (n:Label) ON n.property
        """
        print("Creating Neo4j vector indexes...")
        
        # Node entity embeddings index
        entity_index_query = f"""
        CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
        FOR (n:Node)
        ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.dimension},
                `vector.similarity_function`: '{self.config.similarity_function}'
            }}
        }}
        """
        
        # Relationship embeddings index (stored on Rel relationships)
        rel_index_query = f"""
        CREATE VECTOR INDEX relationship_embedding_index IF NOT EXISTS
        FOR ()-[r:Rel]-()
        ON r.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.config.dimension},
                `vector.similarity_function`: '{self.config.similarity_function}'
            }}
        }}
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                session.run(entity_index_query)
                print("  ✓ Created entity embedding index")
            except Exception as e:
                print(f"  Note: Entity index already exists or Neo4j version < 5.11")
            
            try:
                session.run(rel_index_query)
                print("  ✓ Created relationship embedding index")
            except Exception as e:
                print(f"  Note: Relationship index already exists or Neo4j version < 5.11")
        
        print("Vector index creation complete")
    
    def add_entity_with_embedding(
        self,
        uri: str,
        name: str,
        entity_type: str,
        user: str = "default",
        collection: str = "default",
        context: Optional[str] = None,
        properties: Optional[Dict] = None
    ):
        """
        Add entity with vector embedding
        
        Stores:
        - Entity as Node
        - Embedding as node property (float array)
        - Context and metadata
        """
        # Generate embedding
        embedding = self.embedder.embed_entity(uri, name, entity_type, context)
        
        # Convert to list for Neo4j
        embedding_list = embedding.tolist()
        
        # Store in Neo4j
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (n:Node {uri: $uri, user: $user, collection: $collection})
                SET n.name = $name,
                    n.entity_type = $entity_type,
                    n.embedding = $embedding,
                    n.context = $context,
                    n.properties = $properties
                """,
                uri=uri,
                user=user,
                collection=collection,
                name=name,
                entity_type=entity_type,
                embedding=embedding_list,
                context=context,
                properties=json.dumps(properties) if properties else None
            )
    
    def similarity_search(
        self,
        query_text: str,
        user: str = "default",
        collection: str = "default",
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Semantic similarity search
        
        Process:
        1. Embed query text
        2. Use Neo4j vector index for approximate nearest neighbors
        3. Return top-k most similar entities
        
        Args:
            query_text: Natural language query
            user: User scope
            collection: Collection scope
            limit: Max results
            min_score: Minimum similarity threshold
        
        Returns:
            List of SearchResult ordered by similarity
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query_text)
        query_list = query_embedding.tolist()
        
        # Neo4j vector similarity query
        # Uses HNSW index for fast approximate search
        query = """
        CALL db.index.vector.queryNodes(
            'entity_embedding_index',
            $limit,
            $query_embedding
        ) YIELD node, score
        WHERE node.user = $user AND node.collection = $collection
            AND score >= $min_score
        RETURN
            node.uri AS uri,
            score,
            node.entity_type AS entity_type,
            node.name AS name,
            node.context AS context,
            node.properties AS properties
        ORDER BY score DESC
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                limit=limit,
                query_embedding=query_list,
                user=user,
                collection=collection,
                min_score=min_score
            )
            
            results = []
            for record in result:
                props = json.loads(record['properties']) if record['properties'] else {}
                
                results.append(SearchResult(
                    uri=record['uri'],
                    score=record['score'],
                    entity_type=record.get('entity_type'),
                    properties=props,
                    context=record.get('context')
                ))
            
            return results
    
    def hybrid_search(
        self,
        query_text: str,
        user: str = "default",
        collection: str = "default",
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Hybrid search: Vector similarity + Graph structure
        
        Combines:
        1. Vector similarity (semantic matching)
        2. Graph centrality (important nodes)
        
        Score = vector_weight * similarity + graph_weight * centrality
        
        WHY HYBRID:
        - Vector search finds semantically similar entities
        - Graph metrics find structurally important entities
        - Combination gives best of both worlds
        """
        # Get vector search results
        vector_results = self.similarity_search(
            query_text, user, collection, limit * 2
        )
        
        # Get URIs for graph queries
        uris = [r.uri for r in vector_results]
        
        if not uris:
            return []
        
        # Calculate graph centrality (degree centrality)
        centrality_query = """
        UNWIND $uris AS uri
        MATCH (n:Node {uri: uri, user: $user, collection: $collection})
        OPTIONAL MATCH (n)-[r:Rel]-()
        WITH n, uri, count(r) AS degree
        RETURN uri, degree
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                centrality_query,
                uris=uris,
                user=user,
                collection=collection
            )
            
            # Build centrality map
            centrality_map = {}
            max_degree = 1
            for record in result:
                degree = record['degree'] or 0
                centrality_map[record['uri']] = degree
                max_degree = max(max_degree, degree)
            
            # Normalize centrality scores
            for uri in centrality_map:
                centrality_map[uri] = centrality_map[uri] / max_degree
        
        # Combine scores
        for result in vector_results:
            centrality = centrality_map.get(result.uri, 0.0)
            combined_score = (
                vector_weight * result.score +
                graph_weight * centrality
            )
            result.score = combined_score
        
        # Re-sort by combined score
        vector_results.sort(key=lambda r: r.score, reverse=True)
        
        return vector_results[:limit]
    
    def find_similar_entities(
        self,
        entity_uri: str,
        user: str = "default",
        collection: str = "default",
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find entities similar to a given entity
        
        Uses entity's embedding to find nearest neighbors
        """
        # Get entity embedding
        query = """
        MATCH (n:Node {uri: $uri, user: $user, collection: $collection})
        RETURN n.embedding AS embedding, n.name AS name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                uri=entity_uri,
                user=user,
                collection=collection
            )
            
            record = result.single()
            if not record or not record['embedding']:
                return []
            
            # Use entity's embedding as query
            embedding = record['embedding']
            name = record['name'] or entity_uri
        
        # Find similar entities (exclude self)
        query = """
        CALL db.index.vector.queryNodes(
            'entity_embedding_index',
            $limit_plus_one,
            $embedding
        ) YIELD node, score
        WHERE node.user = $user AND node.collection = $collection
            AND node.uri <> $exclude_uri
        RETURN
            node.uri AS uri,
            score,
            node.entity_type AS entity_type,
            node.name AS name,
            node.context AS context
        ORDER BY score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                limit=limit,
                limit_plus_one=limit + 1,
                embedding=embedding,
                user=user,
                collection=collection,
                exclude_uri=entity_uri
            )
            
            results = []
            for record in result:
                results.append(SearchResult(
                    uri=record['uri'],
                    score=record['score'],
                    entity_type=record.get('entity_type'),
                    context=record.get('context')
                ))
            
            return results


# ================== Vector Aggregation for Multi-Hop Queries ==================

class VectorAggregator:
    """
    Aggregate embeddings for multi-hop graph paths
    
    AGGREGATION STRATEGIES:
    1. Mean: Average all vectors (simple, effective)
    2. Weighted: Weight by edge importance
    3. Attention: Learn which parts matter most
    """
    
    @staticmethod
    def mean_aggregation(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Simple average of embeddings
        
        WHY: Works well for paths of similar importance
        """
        if not embeddings:
            return np.zeros(384)
        
        stacked = np.stack(embeddings)
        mean = np.mean(stacked, axis=0)
        return mean / np.linalg.norm(mean)
    
    @staticmethod
    def weighted_aggregation(
        embeddings: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Weighted average of embeddings
        
        WHY: Emphasize more important entities/relationships
        Use case: Weight by node degree, edge frequency, etc.
        """
        if not embeddings or not weights:
            return np.zeros(384)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted sum
        stacked = np.stack(embeddings)
        weighted = np.sum(stacked * weights[:, None], axis=0)
        
        return weighted / np.linalg.norm(weighted)
    
    @staticmethod
    def max_pooling(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Max pooling across embeddings
        
        WHY: Preserves strongest signals
        """
        if not embeddings:
            return np.zeros(384)
        
        stacked = np.stack(embeddings)
        maxed = np.max(stacked, axis=0)
        return maxed / np.linalg.norm(maxed)
