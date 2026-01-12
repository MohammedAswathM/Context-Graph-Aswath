import json
from pathlib import Path
from typing import Dict, Optional

from .models import Ontology, OntologyMetadata, OntologyClass, OntologyProperty


class OntologyLoader:
    """Load and manage ontologies"""
    
    def __init__(self):
        self.ontologies: Dict[str, Ontology] = {}
    
    def load_from_file(self, file_path: str | Path) -> Ontology:
        """Load ontology from JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {file_path}")
        
        print(f"Loading ontology from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.load_from_dict(data)
    
    def load_from_dict(self, data: dict) -> Ontology:
        """Load ontology from dictionary"""
        # Parse metadata
        metadata = OntologyMetadata(**data['metadata'])
        
        # Parse classes
        classes = {}
        for class_id, class_data in data.get('classes', {}).items():
            classes[class_id] = OntologyClass(**class_data)
        
        # Parse object properties
        object_props = {}
        for prop_id, prop_data in data.get('object_properties', {}).items():
            object_props[prop_id] = OntologyProperty(**prop_data)
        
        # Parse datatype properties
        datatype_props = {}
        for prop_id, prop_data in data.get('datatype_properties', {}).items():
            datatype_props[prop_id] = OntologyProperty(**prop_data)
        
        # Create ontology
        ontology = Ontology(
            metadata=metadata,
            classes=classes,
            object_properties=object_props,
            datatype_properties=datatype_props
        )
        
        # Validate
        issues = ontology.validate_structure()
        if issues:
            print(f"Warning: Ontology validation issues: {issues}")
        
        # Store
        self.ontologies[metadata.ontology_id] = ontology
        
        print(
            f"Loaded ontology '{metadata.ontology_id}': "
            f"{len(classes)} classes, "
            f"{len(object_props)} object properties, "
            f"{len(datatype_props)} datatype properties"
        )
        
        return ontology
    
    def get_ontology(self, ontology_id: str) -> Optional[Ontology]:
        """Get loaded ontology by ID"""
        return self.ontologies.get(ontology_id)
    
    def get_all_ontologies(self) -> Dict[str, Ontology]:
        """Get all loaded ontologies"""
        return self.ontologies.copy()