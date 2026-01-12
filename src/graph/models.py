from pydantic import BaseModel, Field
from typing import Optional


class Value(BaseModel):
    """RDF value (can be URI or literal)"""
    value: str = Field(..., description="The actual value")
    is_uri: bool = Field(default=False, description="True if URI, False if literal")


class Triple(BaseModel):
    """RDF triple (subject-predicate-object)"""
    subject: Value = Field(..., alias="s")
    predicate: Value = Field(..., alias="p")
    object: Value = Field(..., alias="o")
    
    model_config = {"populate_by_name": True}
    
    def __str__(self) -> str:
        return f"<{self.subject.value}> <{self.predicate.value}> <{self.object.value}>"


class TripleMetadata(BaseModel):
    """Metadata for triple batch"""
    user: str = Field(default="default", description="User/tenant ID")
    collection: str = Field(default="default", description="Collection/namespace")
    source: Optional[str] = Field(None, description="Source document")
    timestamp: Optional[str] = Field(None, description="Creation timestamp")


class TriplesBatch(BaseModel):
    """Batch of triples with metadata"""
    triples: list[Triple] = Field(default_factory=list)
    metadata: TripleMetadata = Field(default_factory=TripleMetadata)