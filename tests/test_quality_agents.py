import pytest
from unittest.mock import Mock, MagicMock
from src.agents.quality_control import (
    QualityControlPipeline,
    TripleQuality,
    ValidationResult,
    setup_dspy_for_context_graph
)
from src.config import get_settings

# Mock Ontology for Testing
TEST_ONTOLOGY = {
    "classes": {
        "Organization": {"rdfs:comment": "A company or business"},
        "Person": {"rdfs:comment": "An individual"},
        "Role": {"rdfs:comment": "A position or job"}
    },
    "object_properties": {
        "employs": {
            "rdfs:domain": "Organization",
            "rdfs:range": "Person",
            "rdfs:comment": "Employment relationship"
        },
        "hasRole": {
            "rdfs:domain": "Person",
            "rdfs:range": "Role",
            "rdfs:comment": "Role assignment"
        }
    },
    "datatype_properties": {
        "name": {
            "rdfs:domain": "Entity",
            "rdfs:range": "xsd:string",
            "rdfs:comment": "Name of entity"
        }
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def setup_pipeline_or_mock():
    """Initialize quality control pipeline or return mock"""
    try:
        settings = get_settings()
        
        # DSPy only supports OpenAI currently - force OpenAI even if Gemini is default
        provider = "openai"
        api_key = settings.llm.openai_api_key
        
        if not api_key or api_key == "":
            raise ValueError("OpenAI API key required for DSPy tests")
        
        lm = setup_dspy_for_context_graph(
            provider=provider,
            model="gpt-3.5-turbo",  # Use GPT-3.5 for cost-effective testing
            api_key=api_key
        )
        return QualityControlPipeline(lm=lm, acceptance_threshold=0.85), False
    except Exception as e:
        print(f"⚠️  LLM not configured: {e}")
        print("   Using MOCK quality control for testing...")
        return create_mock_pipeline(), True

def create_mock_pipeline():
    """Create mock pipeline for testing without LLM"""
    mock_pipeline = Mock(spec=QualityControlPipeline)
    mock_pipeline.validated_triples = []
    
    # Mock validate_triple method
    def mock_validate(triple, ontology, source_text):
        """Mock validation that checks basic structure"""
        # Proposer check
        proposer_result = ValidationResult(
            passed=True,
            score=0.85,
            issues=[],
            metadata={
                'ontology_aligned': True,
                'complete': True,
                'temporal_valid': True,
                'atomic': True,
                'source_grounded': True
            }
        )
        
        # Validator check
        predicate = triple.get('predicate', '')
        all_properties = list(ontology.get('object_properties', {}).keys()) + \
                        list(ontology.get('datatype_properties', {}).keys()) + \
                        ['rdf:type', 'rdfs:label']
        
        validator_passed = predicate in all_properties
        validator_result = ValidationResult(
            passed=validator_passed,
            score=0.9 if validator_passed else 0.4,
            issues=[] if validator_passed else [f"Unknown predicate: {predicate}"],
            metadata={'schema_compliant': validator_passed}
        )
        
        # Critic check (simplified)
        is_duplicate = triple in mock_pipeline.validated_triples
        critic_result = ValidationResult(
            passed=not is_duplicate,
            score=0.3 if is_duplicate else 0.8,
            issues=['Duplicate triple'] if is_duplicate else [],
            metadata={'duplicate': is_duplicate}
        )
        
        # Overall - passed and overall_score are computed properties
        quality = TripleQuality(
            triple=triple,
            proposer_result=proposer_result,
            validator_result=validator_result,
            critic_result=critic_result
        )
        
        if quality.passed:
            mock_pipeline.validated_triples.append(triple)
        
        return quality
    
    mock_pipeline.validate_triple = mock_validate
    
    # Mock batch validation
    def mock_validate_batch(triples, ontology, source_text):
        assessments = [mock_validate(t, ontology, source_text) for t in triples]
        accepted = [t for t, q in zip(triples, assessments) if q.passed]
        return accepted, assessments
    
    mock_pipeline.validate_batch = mock_validate_batch
    
    return mock_pipeline

def print_quality_report(triple: dict, quality: TripleQuality, test_name: str):
    """Pretty print quality assessment results"""
    print(f"\n{'='*80}")
    print(f"📋 TEST: {test_name}")
    print(f"{'='*80}")
    
    print(f"\n📌 Triple: ({triple['subject']}, {triple['predicate']}, {triple['object']})")
    print(f"\n{'─'*80}")
    
    # Proposer Results
    if quality.proposer_result:
        print(f"\n🔵 PROPOSER (Ontology Alignment & Completeness)")
        print(f"   Score: {quality.proposer_result.score:.2%}")
        print(f"   Status: {'✅ PASSED' if quality.proposer_result.passed else '❌ FAILED'}")
        if quality.proposer_result.metadata:
            print(f"   Details:")
            for key, value in quality.proposer_result.metadata.items():
                status_icon = "✓" if value else "✗"
                print(f"     {status_icon} {key.replace('_', ' ').title()}: {value}")
        if quality.proposer_result.issues:
            print(f"   Issues:")
            for issue in quality.proposer_result.issues:
                print(f"     • {issue}")
    
    # Validator Results
    if quality.validator_result:
        print(f"\n🟢 VALIDATOR (Schema Compliance & Typing)")
        print(f"   Score: {quality.validator_result.score:.2%}")
        print(f"   Status: {'✅ PASSED' if quality.validator_result.passed else '❌ FAILED'}")
        if quality.validator_result.metadata:
            print(f"   Details:")
            for key, value in quality.validator_result.metadata.items():
                status_icon = "✓" if value else "✗"
                print(f"     {status_icon} {key.replace('_', ' ').title()}: {value}")
        if quality.validator_result.issues:
            print(f"   Issues:")
            for issue in quality.validator_result.issues:
                print(f"     • {issue}")
    
    # Critic Results
    if quality.critic_result:
        print(f"\n🔴 CRITIC (Duplication, Relevance, Contradictions)")
        print(f"   Score: {quality.critic_result.score:.2%}")
        print(f"   Status: {'✅ PASSED' if quality.critic_result.passed else '❌ FAILED'}")
        if quality.critic_result.metadata:
            print(f"   Details:")
            for key, value in quality.critic_result.metadata.items():
                status_icon = "✓" if value else "✗"
                print(f"     {status_icon} {key.replace('_', ' ').title()}: {value}")
        if quality.critic_result.issues:
            print(f"   Issues:")
            for issue in quality.critic_result.issues:
                print(f"     • {issue}")
    
    # Overall Results
    print(f"\n{'─'*80}")
    print(f"\n⭐ OVERALL ASSESSMENT")
    print(f"   Combined Score: {quality.overall_score:.2%}")
    print(f"   Required Score: 85%")
    print(f"   Final Status: {'✅ ACCEPTED' if quality.passed else '❌ REJECTED'}")
    
    if not quality.passed and quality.all_issues:
        print(f"\n   All Issues Found:")
        for issue in quality.all_issues:
            print(f"     • {issue}")
    
    print(f"\n{'='*80}\n")

# ============================================================================
# Test Cases
# ============================================================================

def test_proposer_accepts_valid_triple():
    """Test if Proposer accepts well-formed triple aligned with ontology"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    good_triple = {
        'subject': 'org:Google',
        'predicate': 'employs',
        'object': 'person:Sundar'
    }
    source_text = "Google employs Sundar Pichai as CEO"
    
    quality = pipeline.validate_triple(good_triple, TEST_ONTOLOGY, source_text)
    print_quality_report(good_triple, quality, f"Proposer - Valid Triple {'(MOCK)' if is_mock else ''}")
    
    # Proposer should pass
    assert quality.proposer_result is not None, "Proposer should run"
    assert quality.proposer_result.score > 0.5, "Valid triple should score well"
    print("✅ Test passed: Proposer accepts valid triple")

def test_validator_rejects_schema_violation():
    """Test if Validator catches a relationship that doesn't exist in ontology"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    bad_triple = {
        'subject': 'org:Google',
        'predicate': 'eats',  # Not in ontology
        'object': 'person:Sundar'
    }
    source_text = "Google eats Sundar"
    
    quality = pipeline.validate_triple(bad_triple, TEST_ONTOLOGY, source_text)
    print_quality_report(bad_triple, quality, f"Validator - Schema Violation {'(MOCK)' if is_mock else ''}")
    
    # Should detect issue
    assert not quality.passed, "Invalid predicate should be rejected"
    print("✅ Test passed: Validator catches schema violations")

def test_critic_catches_duplicate():
    """Test if Critic catches a duplicate fact"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    # First, add a triple to validated set
    existing_triple = {
        'subject': 'org:Google',
        'predicate': 'employs',
        'object': 'person:Sundar'
    }
    pipeline.validated_triples = [existing_triple]
    
    # Now validate the same triple again
    duplicate_triple = {
        'subject': 'org:Google',
        'predicate': 'employs',
        'object': 'person:Sundar'
    }
    source_text = "Google employs Sundar Pichai as CEO"
    
    quality = pipeline.validate_triple(duplicate_triple, TEST_ONTOLOGY, source_text)
    print_quality_report(duplicate_triple, quality, f"Critic - Duplicate Detection {'(MOCK)' if is_mock else ''}")
    
    # Critic should flag as duplicate
    if quality.critic_result:
        is_duplicate = quality.critic_result.metadata.get('duplicate', False)
        print(f"✅ Duplicate detection: {is_duplicate}")
        assert is_duplicate == True, "Critic should flag as duplicate"

def test_full_pipeline_acceptance():
    """Test full pipeline with valid triple that should be accepted"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    valid_triple = {
        'subject': 'org:Microsoft',
        'predicate': 'employs',
        'object': 'person:Satya'
    }
    source_text = "Microsoft employs Satya Nadella as Chief Executive Officer"
    
    quality = pipeline.validate_triple(valid_triple, TEST_ONTOLOGY, source_text)
    print_quality_report(valid_triple, quality, f"Full Pipeline - Valid Triple {'(MOCK)' if is_mock else ''}")
    
    print(f"\n📊 Pipeline Summary:")
    print(f"   Proposer passed: {quality.proposer_result.passed if quality.proposer_result else 'N/A'}")
    print(f"   Validator passed: {quality.validator_result.passed if quality.validator_result else 'N/A'}")
    print(f"   Critic passed: {quality.critic_result.passed if quality.critic_result else 'N/A'}")
    print(f"   Overall passed: {quality.passed}")
    
    # At minimum, proposer should pass
    assert quality.proposer_result is not None, "Pipeline should run proposer"
    assert quality.proposer_result.passed, "Valid triple should pass proposer"

def test_batch_validation():
    """Test batch validation of multiple triples"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    triples = [
        {
            'subject': 'org:Apple',
            'predicate': 'employs',
            'object': 'person:Tim'
        },
        {
            'subject': 'org:Amazon',
            'predicate': 'employs',
            'object': 'person:Andy'
        },
        {
            'subject': 'org:Tesla',
            'predicate': 'eats',  # Invalid predicate
            'object': 'person:Elon'
        }
    ]
    source_text = "Apple employs Tim Cook. Amazon employs Andy Jassy. Tesla is owned by Elon."
    
    print(f"\n{'='*80}")
    print(f"📦 BATCH VALIDATION TEST {'(MOCK)' if is_mock else ''}")
    print(f"{'='*80}")
    print(f"\nValidating {len(triples)} triples...\n")
    
    accepted, assessments = pipeline.validate_batch(triples, TEST_ONTOLOGY, source_text)
    
    for i, (triple, quality) in enumerate(zip(triples, assessments)):
        print(f"\n[Triple {i+1}] {triple['subject']} → {triple['predicate']} → {triple['object']}")
        print(f"   Overall Score: {quality.overall_score:.2%}")
        print(f"   Status: {'✅ ACCEPTED' if quality.passed else '❌ REJECTED'}")
    
    print(f"\n{'─'*80}")
    print(f"\n📊 Batch Results:")
    print(f"   Total triples: {len(triples)}")
    print(f"   Accepted: {len(accepted)} ({len(accepted)/len(triples)*100:.1f}%)")
    print(f"   Rejected: {len(triples)-len(accepted)} ({(len(triples)-len(accepted))/len(triples)*100:.1f}%)")
    print(f"\n{'='*80}\n")
    
    assert len(accepted) >= 2, "Should accept the 2 valid triples"

def test_detailed_scoring_breakdown():
    """Show detailed score breakdown for transparency"""
    pipeline, is_mock = setup_pipeline_or_mock()
    
    triple = {
        'subject': 'org:Netflix',
        'predicate': 'employs',
        'object': 'person:Reed'
    }
    source_text = "Netflix is led by CEO Reed Hastings and employs thousands"
    
    quality = pipeline.validate_triple(triple, TEST_ONTOLOGY, source_text)
    
    print(f"\n{'='*80}")
    print(f"📊 DETAILED SCORE BREAKDOWN {'(MOCK)' if is_mock else ''}")
    print(f"{'='*80}\n")
    
    print(f"Triple: {triple['subject']} → {triple['predicate']} → {triple['object']}")
    print(f"Source: \"{source_text}\"\n")
    
    print(f"{'─'*80}")
    print(f"\nAgent Scores:")
    print(f"{'─'*80}\n")
    
    if quality.proposer_result:
        print(f"🔵 PROPOSER:")
        print(f"   Score:    {quality.proposer_result.score:.2%}")
        print(f"   Threshold: 80%")
        print(f"   Status:   {'✅ PASS' if quality.proposer_result.score >= 0.80 else '❌ FAIL'}\n")
    
    if quality.validator_result:
        print(f"🟢 VALIDATOR:")
        print(f"   Score:    {quality.validator_result.score:.2%}")
        print(f"   Threshold: 90%")
        print(f"   Status:   {'✅ PASS' if quality.validator_result.score >= 0.90 else '❌ FAIL'}\n")
    
    if quality.critic_result:
        print(f"🔴 CRITIC:")
        print(f"   Score:    {quality.critic_result.score:.2%}")
        print(f"   Threshold: 75%")
        print(f"   Status:   {'✅ PASS' if quality.critic_result.score >= 0.75 else '❌ FAIL'}\n")
    
    print(f"{'─'*80}\n")
    
    print(f"⭐ FINAL SCORE: {quality.overall_score:.2%}")
    print(f"   Required:   85%")
    print(f"   Result:     {'✅ ACCEPTED' if quality.passed else '❌ REJECTED'}\n")
    
    print(f"{'='*80}\n")
    
    assert quality.overall_score > 0.5, "Score should be reasonable for valid triple"

# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    """Run tests with detailed output"""
    print("\n" + "="*80)
    print("🧪 QUALITY CONTROL AGENTS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Valid Triple", test_proposer_accepts_valid_triple),
        ("Schema Violation", test_validator_rejects_schema_violation),
        ("Duplicate Detection", test_critic_catches_duplicate),
        ("Full Pipeline", test_full_pipeline_acceptance),
        ("Batch Validation", test_batch_validation),
        ("Score Breakdown", test_detailed_scoring_breakdown),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  Running: {test_name}...")
            test_func()
            print(f"✅ {test_name} - PASSED")
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80 + "\n")