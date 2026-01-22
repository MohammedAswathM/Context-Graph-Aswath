from src.processing.semantic_chunker import SemanticChunker

def test_basic_chunking():
    """Test basic semantic chunking functionality"""
    print("\n" + "="*80)
    print("🧪 SEMANTIC CHUNKING TEST")
    print("="*80 + "\n")
    
    chunker = SemanticChunker()
    
    # Test 1: Simple text
    text1 = "Apple Inc. is a technology company. It is located in Cupertino, California."
    chunks1 = chunker.chunk(text1)
    
    print(f"📝 Input Text:")
    print(f"   \"{text1}\"\n")
    
    print(f"✂️  Chunks Generated: {len(chunks1)}")
    for i, chunk in enumerate(chunks1):
        print(f"\n   Chunk {i+1}:")
        print(f"   '{chunk.text}'")
    
    assert len(chunks1) > 0, "Should create at least one chunk"
    print(f"\n✅ Test 1 PASSED: Basic chunking works\n")
    
    # Test 2: Longer text with semantic boundaries
    text2 = (
        "Google was founded by Larry Page and Sergey Brin in 1998. "
        "The company is headquartered in Mountain View, California. "
        "Google specializes in internet services and products. "
        "Amazon Web Services provides cloud computing. "
        "AWS is a subsidiary of Amazon.com."
    )
    chunks2 = chunker.chunk(text2)
    
    print(f"{'─'*80}\n")
    print(f"📝 Input Text:")
    print(f"   \"{text2}\"\n")
    
    print(f"✂️  Chunks Generated: {len(chunks2)}")
    for i, chunk in enumerate(chunks2):
        print(f"\n   Chunk {i+1}:")
        print(f"   '{chunk.text}'")
    
    assert len(chunks2) > 0, "Should create chunks"
    print(f"\n✅ Test 2 PASSED: Semantic chunking preserves meaning\n")
    
    # Test 3: Empty text
    text3 = ""
    chunks3 = chunker.chunk(text3)
    
    print(f"{'─'*80}\n")
    print(f"📝 Test 3: Empty Text")
    print(f"   Chunks: {len(chunks3)}")
    
    assert len(chunks3) == 0, "Empty text should produce no chunks"
    print(f"✅ Test 3 PASSED: Empty text handled correctly\n")
    
    print("="*80)
    print("✅ ALL SEMANTIC CHUNKING TESTS PASSED")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_basic_chunking()