import json
import re
from typing import List, Optional
from ..graph.models import Triple, Value


class ExtractionResult:
    """Parsed extraction result"""
    
    def __init__(self, triples: List[Triple]):
        self.triples = triples
    
    def __len__(self) -> int:
        return len(self.triples)


class ResponseParser:
    """Parse LLM responses into triples"""
    
    # RDF constants
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
    
    @staticmethod
    def parse(response: str, ontology_id: str = "default") -> Optional[ExtractionResult]:
        """
        Parse LLM response into triples.
        Handles various response formats (JSON with/without markdown).
        """
        print("Parsing LLM response")
        
        # Clean response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response:
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
        elif "```" in response:
            response = re.sub(r'```\s*', '', response)
        
        # Try to parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response: {response[:500]}")
            
            # Try to salvage partially valid JSON
            # Look for complete JSON objects before the error
            try:
                # Find last complete object in array
                last_complete = response.rfind('},\n  {')
                if last_complete > 0:
                    truncated = response[:last_complete + 1] + '\n]'
                    data = json.loads(truncated)
                    print(f"Recovered {len(data)} triples from truncated response")
                else:
                    return None
            except:
                return None
        
        if not isinstance(data, list):
            print("Response is not a JSON array")
            return None
        
        # Convert to Triple objects
        triples = []
        
        for item in data:
            try:
                # Create subject
                subject = Value(
                    value=ResponseParser._normalize_uri(item["subject"], ontology_id),
                    is_uri=True
                )
                
                # Create predicate
                predicate = Value(
                    value=ResponseParser._normalize_predicate(item["predicate"]),
                    is_uri=True
                )
                
                # Create object
                is_object_uri = item.get("is_object_uri", True)
                object_value = item["object"]
                
                if is_object_uri:
                    # Object is an entity URI
                    obj = Value(
                        value=ResponseParser._normalize_uri(object_value, ontology_id),
                        is_uri=True
                    )
                else:
                    # Object is a literal
                    obj = Value(
                        value=str(object_value),
                        is_uri=False
                    )
                
                triple = Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj
                )
                
                triples.append(triple)
                
            except KeyError as e:
                print(f"Skipping malformed triple: missing {e}")
                continue
            except Exception as e:
                print(f"Error parsing triple: {e}")
                continue
        
        print(f"Parsed {len(triples)} triples from response")
        return ExtractionResult(triples)
    
    @staticmethod
    def _normalize_uri(uri: str, ontology_id: str) -> str:
        """Normalize entity URI"""
        # If already has scheme, return as-is
        if ":" in uri and not uri.startswith("http"):
            return uri
        
        # If looks like HTTP URL, return as-is
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri
        
        # Otherwise, add ontology prefix
        return f"{ontology_id}:{uri}"
    
    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        """Normalize predicate URI"""
        # Handle special predicates
        if predicate == "rdf:type":
            return ResponseParser.RDF_TYPE
        elif predicate == "rdfs:label":
            return ResponseParser.RDFS_LABEL
        
        # If already full URI, return as-is
        if predicate.startswith("http://") or predicate.startswith("https://"):
            return predicate
        
        # Otherwise return with prefix
        return predicate