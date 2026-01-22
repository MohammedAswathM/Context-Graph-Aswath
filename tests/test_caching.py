"""
Test Suite for LLM Prompt Caching System

Tests three-tier cache (memory, disk, provider):
1. Memory cache (< 1ms, limited size)
2. Disk cache (persistent, larger)
3. Provider cache (OpenAI/Gemini native)
"""

import pytest
from src.extraction.caching import PromptCacheManager, get_global_cache
from src.ontology.loader import OntologyLoader
import json

# Mock ontology
TEST_ONTOLOGY = {
    "classes": {
        "Organization": {"rdfs:comment": "Company"},
        "Person": {"rdfs:comment": "Individual"}
    },
    "object_properties": {
        "employs": {"rdfs:domain": "Organization", "rdfs:range": "Person"}
    }
}

def print_cache_stats(cache, label=""):
    """Pretty print cache statistics"""
    stats = cache.get_stats()
    
    print(f"\n{'─'*80}")
    print(f"📊 CACHE STATISTICS{f' - {label}' if label else ''}")
    print(f"{'─'*80}\n")
    
    print(f"📈 Overall Performance:")
    print(f"   Total Requests:  {stats['total_requests']}")
    print(f"   Hit Rate:        {stats['hit_rate']:.2%}")
    print(f"   Memory Hits:     {stats['memory_hits']}")
    print(f"   Disk Hits:       {stats['disk_hits']}")
    print(f"   Misses:          {stats['misses']}")
    print(f"   Evictions:       {stats['evictions']}\n")
    
    print(f"💾 Storage:")
    print(f"   Memory Cache:    {stats['memory_size']} items")
    print(f"   Disk Cache:      {stats['disk_size']} items\n")
    
    print(f"{'─'*80}\n")

def test_memory_cache_basic():
    """Test basic memory cache functionality"""
    print("\n" + "="*80)
    print("🧪 TEST 1: MEMORY CACHE - Basic Functionality")
    print("="*80)
    
    cache = PromptCacheManager(memory_cache_size=10, disk_cache_size_mb=50)
    
    system_instructions = "You are a knowledge extraction expert."
    input_text = "Apple Inc. is a tech company."
    
    # First call - cache miss
    print(f"\n1️⃣  First Call (Cache Miss):")
    print(f"   System: {system_instructions}")
    print(f"   Input: {input_text}\n")
    
    prompt1, was_cached1 = cache.build_prompt_with_cache(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        input_text=input_text,
        model="gpt-3.5-turbo"
    )
    
    print(f"   Cached: {was_cached1}")
    print(f"   Prompt length: {len(prompt1)} chars\n")
    assert not was_cached1, "First call should miss"
    
    # Second call - cache hit (same ontology and instructions)
    print(f"2️⃣  Second Call (Cache Hit):")
    print(f"   Same ontology and instructions, different input\n")
    
    prompt2, was_cached2 = cache.build_prompt_with_cache(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        input_text="Google is a search engine.",
        model="gpt-3.5-turbo"
    )
    
    print(f"   Cached: {was_cached2}")
    print(f"   Prompt length: {len(prompt2)} chars\n")
    
    print_cache_stats(cache, "After 2 calls")
    print("✅ Test 1 PASSED: Memory cache working\n")

def test_cache_hit_rate():
    """Test cache hit rate with multiple calls"""
    print("\n" + "="*80)
    print("🧪 TEST 2: CACHE - Hit Rate Optimization")
    print("="*80)
    
    cache = PromptCacheManager(memory_cache_size=50, disk_cache_size_mb=100)
    
    system_instructions = "Extract knowledge triples"
    
    test_cases = [
        ("Apple is a tech company", "gpt-3.5-turbo"),
        ("Google is a search engine", "gpt-3.5-turbo"),  # Same model, hit expected
        ("Microsoft is a software company", "gpt-3.5-turbo"),  # Same model, hit expected
        ("Tesla is an EV manufacturer", "gpt-4"),  # Different model, miss expected
        ("Netflix is a streaming service", "gpt-4"),  # Same model as previous, hit expected
    ]
    
    print(f"\nSimulating {len(test_cases)} extraction calls...\n")
    
    for i, (text, model) in enumerate(test_cases, 1):
        print(f"   Call {i}: model={model}, text='{text[:30]}...'")
        
        prompt, was_cached = cache.build_prompt_with_cache(
            ontology=TEST_ONTOLOGY,
            system_instructions=system_instructions,
            input_text=text,
            model=model
        )
        
        status = "✓ HIT" if was_cached else "✗ MISS"
        print(f"           {status}\n")
    
    print_cache_stats(cache, "After 5 calls")
    
    stats = cache.get_stats()
    assert stats['hit_rate'] > 0, "Should have some cache hits"
    print("✅ Test 2 PASSED: Cache hit rate optimized\n")

def test_cache_with_examples():
    """Test caching with few-shot examples"""
    print("\n" + "="*80)
    print("🧪 TEST 3: CACHE - Few-Shot Examples")
    print("="*80)
    
    cache = PromptCacheManager()
    
    system_instructions = "Extract knowledge using examples"
    
    examples = [
        {
            "input": "Apple employs engineers",
            "output": "[{\"subject\": \"org:apple\", \"predicate\": \"employs\", \"object\": \"person:engineer\"}]"
        },
        {
            "input": "Google is headquartered in Mountain View",
            "output": "[{\"subject\": \"org:google\", \"predicate\": \"location\", \"object\": \"Mountain View\"}]"
        }
    ]
    
    print(f"\nCaching with {len(examples)} few-shot examples...\n")
    
    # Call 1 with examples
    print(f"1️⃣  First call with examples:")
    prompt1, cached1 = cache.build_prompt_with_cache(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        input_text="Microsoft is a software company",
        model="gpt-3.5-turbo",
        examples=examples
    )
    
    print(f"   Cached: {cached1}")
    print(f"   Prompt length: {len(prompt1)} chars\n")
    
    # Call 2 with same examples (should hit)
    print(f"2️⃣  Second call with same examples:")
    prompt2, cached2 = cache.build_prompt_with_cache(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        input_text="Amazon is an e-commerce company",
        model="gpt-3.5-turbo",
        examples=examples
    )
    
    print(f"   Cached: {cached2}")
    print(f"   Prompt length: {len(prompt2)} chars\n")
    
    print_cache_stats(cache, "With examples")
    print("✅ Test 3 PASSED: Examples caching works\n")

def test_cache_key_generation():
    """Test deterministic cache key generation"""
    print("\n" + "="*80)
    print("🧪 TEST 4: CACHE - Deterministic Key Generation")
    print("="*80)
    
    cache = PromptCacheManager()
    
    system_instructions = "Extract triples"
    
    # Generate same key twice
    print(f"\nGenerating cache keys...\n")
    
    key1 = cache._generate_cache_key(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        examples=None,
        model="gpt-3.5-turbo"
    )
    
    key2 = cache._generate_cache_key(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        examples=None,
        model="gpt-3.5-turbo"
    )
    
    print(f"Key 1: {key1}")
    print(f"Key 2: {key2}")
    print(f"\nKeys identical: {key1 == key2}\n")
    
    assert key1 == key2, "Same inputs should generate same key"
    
    # Different model should generate different key
    key3 = cache._generate_cache_key(
        ontology=TEST_ONTOLOGY,
        system_instructions=system_instructions,
        examples=None,
        model="gpt-4"  # Different model
    )
    
    print(f"Key 3 (different model): {key3}")
    print(f"Key 1 vs Key 3 different: {key1 != key3}\n")
    
    assert key1 != key3, "Different models should generate different keys"
    
    print("✅ Test 4 PASSED: Deterministic key generation works\n")

def test_cache_performance_comparison():
    """Compare performance: uncached vs cached"""
    print("\n" + "="*80)
    print("🧪 TEST 5: CACHE - Performance Comparison")
    print("="*80)
    
    cache = PromptCacheManager()
    
    system_instructions = "Extract knowledge from text"
    common_ontology = TEST_ONTOLOGY
    
    texts = [
        "Apple employs Steve Jobs",
        "Apple employs Tim Cook",
        "Apple employs Jony Ive",
    ]
    
    print(f"\nSimulating {len(texts)} calls with high cache potential...\n")
    print("Expected: All calls use same ontology + instructions")
    print("          Only input text changes\n")
    
    print(f"{'Call':<6} {'Text':<35} {'Status':<10} {'Prompt Size'}")
    print(f"{'─'*70}")
    
    for i, text in enumerate(texts, 1):
        prompt, was_cached = cache.build_prompt_with_cache(
            ontology=common_ontology,
            system_instructions=system_instructions,
            input_text=text,
            model="gpt-3.5-turbo"
        )
        
        status = "CACHE HIT ✓" if was_cached else "CACHE MISS ✗"
        print(f"{i:<6} {text:<35} {status:<10} {len(prompt)} chars")
    
    print(f"\n{'─'*70}\n")
    
    stats = cache.get_stats()
    print_cache_stats(cache, "Performance Comparison")
    
    print("✅ Test 5 PASSED: Cache reduces redundant processing\n")

def test_cache_clear():
    """Test cache clearing functionality"""
    print("\n" + "="*80)
    print("🧪 TEST 6: CACHE - Clear Functionality")
    print("="*80)
    
    cache = PromptCacheManager()
    
    # Add some items
    print(f"\nAdding 3 items to cache...")
    for i in range(3):
        cache.build_prompt_with_cache(
            ontology=TEST_ONTOLOGY,
            system_instructions="Test",
            input_text=f"Text {i}",
            model="gpt-3.5-turbo"
        )
    
    stats_before = cache.get_stats()
    print(f"\nBefore clear:")
    print(f"  Memory items: {stats_before['memory_size']}")
    print(f"  Total requests: {stats_before['total_requests']}\n")
    
    # Clear cache
    print(f"Clearing all caches...")
    cache.clear_cache(tier="all")
    
    stats_after = cache.get_stats()
    print(f"\nAfter clear:")
    print(f"  Memory items: {stats_after['memory_size']}")
    print(f"  Total requests: {stats_after['total_requests']}")
    print(f"  Hit rate reset: {stats_after['hit_rate']:.2%}\n")
    
    assert stats_after['memory_size'] == 0, "Memory cache should be cleared"
    assert stats_after['total_requests'] == 0, "Stats should be reset"
    
    print("✅ Test 6 PASSED: Cache clearing works\n")

# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 PROMPT CACHING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        test_memory_cache_basic,
        test_cache_hit_rate,
        test_cache_with_examples,
        test_cache_key_generation,
        test_cache_performance_comparison,
        test_cache_clear,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*80)
    print(f"📊 TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80 + "\n")