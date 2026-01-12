from jinja2 import Template
from ..ontology.models import Ontology


# Main extraction prompt template
EXTRACTION_PROMPT_TEMPLATE = Template("""
You are a knowledge extraction expert. Extract structured triples from text using ONLY the provided ontology elements.

## Ontology Classes:

{% for class_id, class_def in classes.items() %}
- **{{class_id}}**{% if class_def.subclass_of %} (subclass of {{class_def.subclass_of}}){% endif %}{% if class_def.comment %}: {{class_def.comment}}{% endif %}
{% endfor %}

## Object Properties (connect entities):

{% for prop_id, prop_def in object_properties.items() %}
- **{{prop_id}}**{% if prop_def.domain and prop_def.range %} ({{prop_def.domain}} → {{prop_def.range}}){% endif %}{% if prop_def.comment %}: {{prop_def.comment}}{% endif %}
{% endfor %}

## Datatype Properties (entity attributes):

{% for prop_id, prop_def in datatype_properties.items() %}
- **{{prop_id}}**{% if prop_def.domain and prop_def.range %} ({{prop_def.domain}} → {{prop_def.range}}){% endif %}{% if prop_def.comment %}: {{prop_def.comment}}{% endif %}
{% endfor %}

## Text to Analyze:

{{text}}

## Extraction Rules:

1. Only use classes defined above for entity types
2. Only use properties defined above for relationships and attributes
3. Respect domain and range constraints where specified
4. For class instances, use `rdf:type` as the predicate
5. Include `rdfs:label` for new entities to provide human-readable names
6. Extract all relevant triples that can be inferred from the text
7. Use entity URIs or meaningful identifiers as subjects/objects

## Output Format:

Return ONLY a valid JSON array (no markdown, no code blocks) containing objects with these fields:
- "subject": the subject entity (URI or identifier)
- "predicate": the property (from ontology or rdf:type/rdfs:label)
- "object": the object entity or literal value
- "is_object_uri": true if object is an entity, false if literal

## Example Output:

[
  {"subject": "org:acme_corp", "predicate": "rdf:type", "object": "Organization", "is_object_uri": true},
  {"subject": "org:acme_corp", "predicate": "rdfs:label", "object": "ACME Corporation", "is_object_uri": false},
  {"subject": "org:acme_corp", "predicate": "employs", "object": "person:john_doe", "is_object_uri": true},
  {"subject": "person:john_doe", "predicate": "rdf:type", "object": "Person", "is_object_uri": true},
  {"subject": "person:john_doe", "predicate": "rdfs:label", "object": "John Doe", "is_object_uri": false}
]

Now extract triples from the text above.
""")


class PromptBuilder:
    """Build extraction prompts from ontology and text"""
    
    @staticmethod
    def build_extraction_prompt(ontology: Ontology, text: str) -> str:
        """Build extraction prompt"""
        # Prepare ontology data for template
        classes = {}
        for class_id, cls in ontology.classes.items():
            classes[class_id] = {
                "subclass_of": cls.subclass_of,
                "comment": cls.comment
            }
        
        object_properties = {}
        for prop_id, prop in ontology.object_properties.items():
            object_properties[prop_id] = {
                "domain": prop.domain,
                "range": prop.range,
                "comment": prop.comment
            }
        
        datatype_properties = {}
        for prop_id, prop in ontology.datatype_properties.items():
            datatype_properties[prop_id] = {
                "domain": prop.domain,
                "range": prop.range,
                "comment": prop.comment
            }
        
        # Render template
        prompt = EXTRACTION_PROMPT_TEMPLATE.render(
            classes=classes,
            object_properties=object_properties,
            datatype_properties=datatype_properties,
            text=text
        )
        
        return prompt