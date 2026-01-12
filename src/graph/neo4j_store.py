from neo4j import GraphDatabase, Driver
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from ..config import Neo4jSettings
from .models import Triple, TripleMetadata, TriplesBatch, Value


class Neo4jStore:
    """
    Neo4j graph storage for RDF triples.
    
    Graph Model:
    - (:Node {uri, user, collection}) - Entity nodes
    - (:Literal {value, user, collection}) - Literal values
    - [:Rel {uri, user, collection}] - Relationships
    """
    
    def __init__(self, settings: Neo4jSettings):
        self.settings = settings
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        print(f"Connecting to Neo4j at {self.settings.uri}")
        
        self.driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.username, self.settings.password),
            max_connection_pool_size=self.settings.max_connection_pool_size,
            connection_timeout=self.settings.connection_timeout
        )
        
        # Verify connectivity
        self.driver.verify_connectivity()
        print("Successfully connected to Neo4j")
        
        # Ensure database exists (for Neo4j Enterprise/Desktop)
        self._ensure_database()
        
        # Create indexes
        self._create_indexes()
    
    def _ensure_database(self):
        """Ensure database exists (only for Neo4j Enterprise/Aura)"""
        try:
            # Try to create database if it doesn't exist (Enterprise/Aura only)
            with self.driver.session(database="system") as session:
                result = session.run(
                    "SHOW DATABASES WHERE name = $db_name",
                    db_name=self.settings.database
                )
                if not list(result):
                    # Database doesn't exist, try to create it
                    print(f"Creating database '{self.settings.database}'...")
                    session.run(f"CREATE DATABASE {self.settings.database} IF NOT EXISTS")
                    print(f"Database '{self.settings.database}' created")
        except Exception as e:
            # Silently fail for Community Edition (doesn't support system database)
            # Community Edition only has 'neo4j' database by default
            print(f"Note: Using default database (Community Edition detected)")
    
    def _create_indexes(self):
        """Create performance indexes"""
        print("Creating Neo4j indexes...")
        
        indexes = [
            # Node indexes
            "CREATE INDEX node_uri IF NOT EXISTS FOR (n:Node) ON (n.uri)",
            "CREATE INDEX node_collection IF NOT EXISTS FOR (n:Node) ON (n.user, n.collection)",
            "CREATE INDEX node_user_collection_uri IF NOT EXISTS FOR (n:Node) ON (n.user, n.collection, n.uri)",
            
            # Literal indexes
            "CREATE INDEX literal_value IF NOT EXISTS FOR (l:Literal) ON (l.value)",
            "CREATE INDEX literal_collection IF NOT EXISTS FOR (l:Literal) ON (l.user, l.collection)",
            
            # Relationship indexes
            "CREATE INDEX rel_uri IF NOT EXISTS FOR ()-[r:Rel]-() ON (r.uri)",
            "CREATE INDEX rel_user IF NOT EXISTS FOR ()-[r:Rel]-() ON (r.user)",
            "CREATE INDEX rel_collection IF NOT EXISTS FOR ()-[r:Rel]-() ON (r.collection)",
            
            # Collection metadata
            "CREATE INDEX collection_meta IF NOT EXISTS FOR (c:CollectionMetadata) ON (c.user, c.collection)",
        ]
        
        with self.driver.session(database=self.settings.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    pass  # Index may already exist
        
        print("Index creation complete")
    
    def create_collection(
        self,
        user: str,
        collection: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a collection (namespace for triples)"""
        print(f"Creating collection {user}/{collection}")
        
        # Serialize metadata to JSON string (Neo4j doesn't support nested dicts)
        metadata_json = json.dumps(metadata or {})
        
        with self.driver.session(database=self.settings.database) as session:
            session.run(
                """
                MERGE (c:CollectionMetadata {user: $user, collection: $collection})
                SET c.created_at = $created_at,
                    c.metadata = $metadata
                """,
                user=user,
                collection=collection,
                created_at=datetime.utcnow().isoformat(),
                metadata=metadata_json
            )
    
    def collection_exists(self, user: str, collection: str) -> bool:
        """Check if collection exists"""
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (c:CollectionMetadata {user: $user, collection: $collection})
                RETURN c LIMIT 1
                """,
                user=user,
                collection=collection
            )
            return bool(list(result))
    
    def add_triples_batch(self, batch: TriplesBatch):
        """Add multiple triples in a single transaction"""
        user = batch.metadata.user
        collection = batch.metadata.collection
        
        if not self.collection_exists(user, collection):
            self.create_collection(user, collection)
        
        print(f"Adding {len(batch.triples)} triples to {user}/{collection}")
        
        with self.driver.session(database=self.settings.database) as session:
            with session.begin_transaction() as tx:
                for triple in batch.triples:
                    # Create subject
                    tx.run(
                        "MERGE (n:Node {uri: $uri, user: $user, collection: $collection})",
                        uri=triple.subject.value, user=user, collection=collection
                    )
                    
                    if triple.object.is_uri:
                        # Create object node
                        tx.run(
                            "MERGE (n:Node {uri: $uri, user: $user, collection: $collection})",
                            uri=triple.object.value, user=user, collection=collection
                        )
                        # Create relationship
                        tx.run(
                            """
                            MATCH (src:Node {uri: $src, user: $user, collection: $collection})
                            MATCH (dest:Node {uri: $dest, user: $user, collection: $collection})
                            MERGE (src)-[:Rel {uri: $uri, user: $user, collection: $collection}]->(dest)
                            """,
                            src=triple.subject.value,
                            dest=triple.object.value,
                            uri=triple.predicate.value,
                            user=user, collection=collection
                        )
                    else:
                        # Create literal
                        tx.run(
                            "MERGE (l:Literal {value: $value, user: $user, collection: $collection})",
                            value=triple.object.value, user=user, collection=collection
                        )
                        # Create relationship
                        tx.run(
                            """
                            MATCH (src:Node {uri: $src, user: $user, collection: $collection})
                            MATCH (dest:Literal {value: $dest, user: $user, collection: $collection})
                            MERGE (src)-[:Rel {uri: $uri, user: $user, collection: $collection}]->(dest)
                            """,
                            src=triple.subject.value,
                            dest=triple.object.value,
                            uri=triple.predicate.value,
                            user=user, collection=collection
                        )
                
                tx.commit()
        
        print(f"Successfully added {len(batch.triples)} triples")
    
    def query_entity(
        self,
        entity_uri: str,
        user: str = "default",
        collection: str = "default",
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Query entity and its relationships"""
        # Validate max_depth to prevent injection
        max_depth = max(1, min(int(max_depth), 10))  # Limit between 1 and 10
        
        with self.driver.session(database=self.settings.database) as session:
            # Use f-string for max_depth since it's a structural parameter
            query = f"""
                MATCH path = (e:Node {{uri: $uri, user: $user, collection: $collection}})
                             -[:Rel*1..{max_depth}]-(related)
                RETURN e, relationships(path) AS rels, nodes(path) AS nodes
                LIMIT 100
                """
            result = session.run(
                query,
                uri=entity_uri,
                user=user,
                collection=collection
            )
            return [record.data() for record in result]
    
    def get_all_entities(
        self,
        user: str = "default",
        collection: str = "default",
        limit: int = 100
    ) -> List[str]:
        """Get all entity URIs in a collection"""
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (n:Node {user: $user, collection: $collection})
                RETURN n.uri AS uri
                LIMIT $limit
                """,
                user=user,
                collection=collection,
                limit=limit
            )
            return [record["uri"] for record in result]
    
    def delete_collection(self, user: str, collection: str):
        """Delete all data for a collection"""
        print(f"Deleting collection {user}/{collection}")
        
        with self.driver.session(database=self.settings.database) as session:
            # Delete nodes
            session.run(
                "MATCH (n:Node {user: $user, collection: $collection}) DETACH DELETE n",
                user=user, collection=collection
            )
            # Delete literals
            session.run(
                "MATCH (l:Literal {user: $user, collection: $collection}) DETACH DELETE l",
                user=user, collection=collection
            )
            # Delete metadata
            session.run(
                "MATCH (c:CollectionMetadata {user: $user, collection: $collection}) DELETE c",
                user=user, collection=collection
            )
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")