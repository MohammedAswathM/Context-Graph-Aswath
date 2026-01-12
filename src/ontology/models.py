from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class Label(BaseModel):
    """Multi-language label"""
    value: str = Field(..., alias="@value")
    language: str = Field(default="en", alias="@language")
    
    model_config = {"populate_by_name": True}


class OntologyClass(BaseModel):
    """Represents an OWL Class"""
    uri: str = Field(..., description="Full URI of the class")
    type: str = Field(default="owl:Class", description="Type of ontology element")
    labels: List[Label] = Field(default_factory=list, alias="rdfs:label")
    comment: Optional[str] = Field(None, alias="rdfs:comment")
    subclass_of: Optional[str] = Field(None, alias="rdfs:subClassOf")
    
    model_config = {"populate_by_name": True}
    
    @property
    def primary_label(self) -> str:
        """Get primary English label"""
        for label in self.labels:
            if label.language == "en":
                return label.value
        return self.labels[0].value if self.labels else self.uri.split(':')[-1]


class OntologyProperty(BaseModel):
    """Represents an OWL Property"""
    uri: str = Field(..., description="Full URI of the property")
    type: str = Field(..., description="owl:ObjectProperty or owl:DatatypeProperty")
    labels: List[Label] = Field(default_factory=list, alias="rdfs:label")
    comment: Optional[str] = Field(None, alias="rdfs:comment")
    domain: Optional[str] = Field(None, alias="rdfs:domain")
    range: Optional[str] = Field(None, alias="rdfs:range")
    
    model_config = {"populate_by_name": True}
    
    @property
    def primary_label(self) -> str:
        """Get primary English label"""
        for label in self.labels:
            if label.language == "en":
                return label.value
        return self.labels[0].value if self.labels else self.uri.split(':')[-1]


class OntologyMetadata(BaseModel):
    """Ontology metadata"""
    ontology_id: str = Field(..., description="Unique ontology identifier")
    version: str = Field(default="1.0.0", description="Ontology version")
    description: Optional[str] = Field(None, description="Human-readable description")
    created: Optional[str] = Field(None, description="Creation date")


class Ontology(BaseModel):
    """Complete ontology definition"""
    metadata: OntologyMetadata
    classes: Dict[str, OntologyClass] = Field(default_factory=dict)
    object_properties: Dict[str, OntologyProperty] = Field(default_factory=dict)
    datatype_properties: Dict[str, OntologyProperty] = Field(default_factory=dict)
    
    def get_class(self, class_id: str) -> Optional[OntologyClass]:
        """Get class by ID"""
        return self.classes.get(class_id)
    
    def get_property(self, prop_id: str) -> Optional[OntologyProperty]:
        """Get property (object or datatype) by ID"""
        prop = self.object_properties.get(prop_id)
        if prop is None:
            prop = self.datatype_properties.get(prop_id)
        return prop
    
    def validate_structure(self) -> List[str]:
        """Validate ontology and return list of issues"""
        issues = []
        
        # Check for circular inheritance
        for class_id, cls in self.classes.items():
            visited = set()
            current = class_id
            while current:
                if current in visited:
                    issues.append(f"Circular inheritance for class {class_id}")
                    break
                visited.add(current)
                parent_cls = self.get_class(current)
                current = parent_cls.subclass_of if parent_cls else None
        
        # Validate property domains/ranges
        for prop_id, prop in {**self.object_properties, **self.datatype_properties}.items():
            if prop.domain and prop.domain not in self.classes:
                issues.append(f"Property {prop_id} has unknown domain {prop.domain}")
            
            if prop.type == "owl:ObjectProperty" and prop.range:
                if prop.range not in self.classes:
                    issues.append(f"Property {prop_id} has unknown range {prop.range}")
        
        return issues