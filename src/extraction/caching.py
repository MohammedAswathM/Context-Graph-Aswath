import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from diskcache import Cache
from cachetools import LRUCache
import os


class PromptCacheManager:
    """
    Three-tier caching system for LLM prompts:
    1. Memory cache (ultra-fast, limited size)
    2. Disk cache (persistent, larger capacity)
    3. Provider cache (OpenAI/Gemini native caching)
    """
    
    def __init__(
        self,
        memory_cache_size: int = 100,
        disk_cache_size_mb: int = 500,
        cache_dir: str = "cache/llm_prompts"
    ):
        # Memory cache (LRU)
        self.memory_cache = LRUCache(maxsize=memory_cache_size)
        
        # Disk cache (persistent)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.disk_cache = Cache(cache_dir, size_limit=disk_cache_size_mb * 1024 * 1024)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _generate_cache_key(
        self,
        ontology: Dict[str, Any],
        system_instructions: str,
        examples: Optional[list] = None,
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """
        Generate deterministic cache key from STATIC prompt components.
        
        CRITICAL FIX: Only hash ontology + system_instructions + examples + model.
        DO NOT include input_text (changes every request).
        
        Cache Key Components:
        ✅ Ontology structure (defines what to extract)
        ✅ System instructions (how to extract)
        ✅ Examples (few-shot learning)
        ✅ Model (different models need different prompts)
        ❌ Input text (DYNAMIC - unique per request)
        """
        # Create stable representation
        cache_data = {
            'ontology_id': ontology.get('metadata', {}).get('ontology_id', 'unknown'),
            'ontology_classes': sorted(ontology.get('classes', {}).keys()),
            'ontology_properties': sorted(
                list(ontology.get('object_properties', {}).keys()) +
                list(ontology.get('datatype_properties', {}).keys())
            ),
            'system_instructions': system_instructions,
            'model': model,
            'examples': examples if examples else []
        }
        
        # Generate hash
        cache_json = json.dumps(cache_data, sort_keys=True)
        cache_key = hashlib.sha256(cache_json.encode()).hexdigest()
        
        return cache_key
    
    def build_prompt_with_cache(
        self,
        ontology: Dict[str, Any],
        system_instructions: str,
        input_text: str,  # This is NOT used in cache key
        model: str = "gpt-3.5-turbo",
        examples: Optional[list] = None
    ) -> Tuple[str, bool]:
        """
        Build prompt with three-tier caching.
        
        Returns:
            (prompt_template, was_cached): Tuple of prompt and cache hit status
        """
        self.stats['total_requests'] += 1
        
        # Generate cache key (WITHOUT input_text)
        cache_key = self._generate_cache_key(
            ontology=ontology,
            system_instructions=system_instructions,
            examples=examples,
            model=model
        )
        
        # Check memory cache
        if cache_key in self.memory_cache:
            self.stats['memory_hits'] += 1
            prompt_template = self.memory_cache[cache_key]
            # Inject dynamic input text into cached template
            final_prompt = prompt_template.replace("{INPUT_TEXT_PLACEHOLDER}", input_text)
            return final_prompt, True
        
        # Check disk cache
        cached_template = self.disk_cache.get(cache_key)
        if cached_template is not None:
            self.stats['disk_hits'] += 1
            # Promote to memory cache
            self.memory_cache[cache_key] = cached_template
            # Inject dynamic input text
            final_prompt = cached_template.replace("{INPUT_TEXT_PLACEHOLDER}", input_text)
            return final_prompt, True
        
        # Cache miss - build new prompt template
        self.stats['misses'] += 1
        prompt_template = self._build_prompt_template(
            ontology=ontology,
            system_instructions=system_instructions,
            examples=examples,
            model=model
        )
        
        # Store template in both caches
        self.memory_cache[cache_key] = prompt_template
        self.disk_cache.set(cache_key, prompt_template)
        
        # Inject input text into template
        final_prompt = prompt_template.replace("{INPUT_TEXT_PLACEHOLDER}", input_text)
        
        return final_prompt, False
    
    def _build_prompt_template(
        self,
        ontology: Dict[str, Any],
        system_instructions: str,
        examples: Optional[list] = None,
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """
        Build prompt TEMPLATE with placeholder for dynamic input.
        
        Template Structure:
        1. System instructions (static)
        2. Ontology definitions (static)
        3. Examples (static)
        4. {INPUT_TEXT_PLACEHOLDER} (dynamic - replaced per request)
        5. Output format instructions (static)
        """
        from jinja2 import Template
        
        # Build ontology section
        ontology_section = self._format_ontology(ontology)
        
        # Build examples section
        examples_section = ""
        if examples:
            examples_section = "\n## Examples:\n\n"
            for i, example in enumerate(examples, 1):
                examples_section += f"Example {i}:\n"
                examples_section += f"Input: {example.get('input', '')}\n"
                examples_section += f"Output: {example.get('output', '')}\n\n"
        
        # Build complete template
        template = f"""
{system_instructions}

{ontology_section}

{examples_section}

## Text to Analyze:

{{INPUT_TEXT_PLACEHOLDER}}

## Output Format:

Return ONLY a valid JSON array with objects containing:
- "subject": entity URI or identifier
- "predicate": property from ontology
- "object": entity or literal value
- "is_object_uri": true if object is entity, false if literal

Extract all relevant triples from the text above.
"""
        
        return template.strip()
    
    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Format ontology for prompt"""
        sections = []
        
        # Classes
        classes = ontology.get('classes', {})
        if classes:
            sections.append("## Ontology Classes:\n")
            for class_id, class_def in classes.items():
                comment = class_def.get('rdfs:comment', '')
                sections.append(f"- **{class_id}**: {comment}")
        
        # Object properties
        obj_props = ontology.get('object_properties', {})
        if obj_props:
            sections.append("\n## Object Properties (Relationships):\n")
            for prop_id, prop_def in obj_props.items():
                domain = prop_def.get('rdfs:domain', '')
                range_val = prop_def.get('rdfs:range', '')
                comment = prop_def.get('rdfs:comment', '')
                sections.append(f"- **{prop_id}** ({domain} → {range_val}): {comment}")
        
        # Datatype properties
        data_props = ontology.get('datatype_properties', {})
        if data_props:
            sections.append("\n## Datatype Properties (Attributes):\n")
            for prop_id, prop_def in data_props.items():
                comment = prop_def.get('rdfs:comment', '')
                sections.append(f"- **{prop_id}**: {comment}")
        
        return "\n".join(sections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats['total_requests'] > 0:
            total_hits = self.stats['memory_hits'] + self.stats['disk_hits']
            hit_rate = total_hits / self.stats['total_requests']
        
        return {
            'total_requests': self.stats['total_requests'],
            'memory_hits': self.stats['memory_hits'],
            'disk_hits': self.stats['disk_hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'memory_size': len(self.memory_cache),
            'disk_size': len(self.disk_cache),
            'evictions': self.stats['evictions']
        }
    
    def clear_cache(self, tier: str = "all"):
        """
        Clear cache tiers.
        
        Args:
            tier: 'memory', 'disk', or 'all'
        """
        if tier in ['memory', 'all']:
            self.memory_cache.clear()
        
        if tier in ['disk', 'all']:
            self.disk_cache.clear()
        
        if tier == 'all':
            self.stats = {
                'total_requests': 0,
                'memory_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'evictions': 0
            }


# Global cache instance
_global_cache: Optional[PromptCacheManager] = None


def get_global_cache() -> PromptCacheManager:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PromptCacheManager()
    return _global_cache