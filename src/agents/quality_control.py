"""
DSPy-Based Quality Control Agents

Three-stage validation pipeline:
1. Proposer: Validates extraction quality before acceptance
2. Validator: Enforces schema and typing compliance
3. Critic: Detects duplicates, ambiguity, contradictions

WHY DSPy INSTEAD OF RAW PROMPTS:
- Automatic prompt optimization through compilation
- Systematic evaluation and improvement
- Type-safe signatures with validation
- Better than manual prompt engineering
"""

import dspy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ValidationResult:
    """Result of agent validation"""
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class TripleQuality:
    """Quality assessment for a triple"""
    triple: Dict  # The triple being assessed
    proposer_result: Optional[ValidationResult] = None
    validator_result: Optional[ValidationResult] = None
    critic_result: Optional[ValidationResult] = None
    
    @property
    def overall_score(self) -> float:
        """Combined quality score"""
        scores = [
            r.score for r in [
                self.proposer_result,
                self.validator_result,
                self.critic_result
            ] if r is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def passed(self) -> bool:
        """Did triple pass all checks?"""
        results = [
            self.proposer_result,
            self.validator_result,
            self.critic_result
        ]
        return all(r.passed for r in results if r is not None)
    
    @property
    def all_issues(self) -> List[str]:
        """Collect all issues from all agents"""
        issues = []
        for result in [self.proposer_result, self.validator_result, self.critic_result]:
            if result:
                issues.extend(result.issues)
        return issues


# ================== DSPy Signatures ==================

class ProposerSignature(dspy.Signature):
    """Validate ontological alignment and completeness"""
    ontology_definition: str = dspy.InputField(desc="Ontology classes and properties")
    triple: str = dspy.InputField(desc="Triple to validate: (subject, predicate, object)")
    source_text: str = dspy.InputField(desc="Original text excerpt")
    
    # Outputs
    ontology_aligned: bool = dspy.OutputField(desc="Does triple align with ontology?")
    is_complete: bool = dspy.OutputField(desc="Are all essential elements present?")
    has_temporal_context: bool = dspy.OutputField(desc="Is temporal info preserved if present?")
    is_atomic: bool = dspy.OutputField(desc="Does triple represent single fact?")
    has_source_attribution: bool = dspy.OutputField(desc="Can fact be traced to source?")
    
    score: float = dspy.OutputField(desc="Quality score 0.0-1.0")
    issues: str = dspy.OutputField(desc="Comma-separated list of issues (or 'none')")


class ValidatorSignature(dspy.Signature):
    """Enforce schema compliance and typing"""
    ontology_schema: str = dspy.InputField(desc="Schema definitions with types")
    triple: str = dspy.InputField(desc="Triple to validate")
    
    # Outputs
    schema_compliant: bool = dspy.OutputField(desc="Matches schema structure?")
    correct_data_types: bool = dspy.OutputField(desc="Subject/predicate/object types valid?")
    constraints_satisfied: bool = dspy.OutputField(desc="Domain/range constraints met?")
    
    score: float = dspy.OutputField(desc="Compliance score 0.0-1.0")
    issues: str = dspy.OutputField(desc="Comma-separated list of issues (or 'none')")


class CriticSignature(dspy.Signature):
    """Detect quality issues and contradictions"""
    triple: str = dspy.InputField(desc="Triple to critique")
    existing_triples: str = dspy.InputField(desc="Previously validated triples")
    source_text: str = dspy.InputField(desc="Original text context")
    
    # Outputs
    is_duplicate: bool = dspy.OutputField(desc="Duplicate of existing triple?")
    is_relevant: bool = dspy.OutputField(desc="Relevant to domain?")
    is_unambiguous: bool = dspy.OutputField(desc="Entities clearly identified?")
    no_contradictions: bool = dspy.OutputField(desc="Consistent with existing knowledge?")
    
    score: float = dspy.OutputField(desc="Quality score 0.0-1.0")
    issues: str = dspy.OutputField(desc="Comma-separated list of issues (or 'none')")


# ================== DSPy Agent Modules ==================

class ProposerAgent(dspy.Module):
    """
    Validates ontological alignment and completeness
    
    CHECKS:
    - Ontology Alignment: Does triple fit defined schema?
    - Completeness: All necessary information present?
    - Temporal Context: Time info preserved if mentioned?
    - Atomicity: Single fact, not compound?
    - Source Attribution: Can trace back to source?
    """
    
    def __init__(self):
        super().__init__()
        self.propose = dspy.ChainOfThought(ProposerSignature)
    
    def forward(
        self,
        triple: Dict,
        ontology: Dict,
        source_text: str
    ) -> ValidationResult:
        """Validate proposed triple"""
        
        # Format inputs
        ontology_def = self._format_ontology(ontology)
        triple_str = f"({triple['subject']}, {triple['predicate']}, {triple['object']})"
        
        # Run DSPy reasoning
        result = self.propose(
            ontology_definition=ontology_def,
            triple=triple_str,
            source_text=source_text
        )
        
        # Calculate score
        checks = [
            result.ontology_aligned,
            result.is_complete,
            result.has_temporal_context,
            result.is_atomic,
            result.has_source_attribution
        ]
        passed_checks = sum(checks)
        score = passed_checks / len(checks)
        
        # Parse issues
        issues = self._parse_issues(result.issues)
        
        return ValidationResult(
            passed=score >= 0.8,  # Require 80% threshold
            score=score,
            issues=issues,
            metadata={
                'ontology_aligned': result.ontology_aligned,
                'complete': result.is_complete,
                'temporal': result.has_temporal_context,
                'atomic': result.is_atomic,
                'sourced': result.has_source_attribution
            }
        )
    
    def _format_ontology(self, ontology: Dict) -> str:
        """Format ontology for DSPy"""
        parts = []
        
        if 'classes' in ontology:
            classes = ', '.join(ontology['classes'].keys())
            parts.append(f"Classes: {classes}")
        
        if 'object_properties' in ontology:
            props = ', '.join(ontology['object_properties'].keys())
            parts.append(f"Relations: {props}")
        
        return ' | '.join(parts)
    
    def _parse_issues(self, issues_str: str) -> List[str]:
        """Parse comma-separated issues"""
        if issues_str.lower() in ['none', 'n/a', '']:
            return []
        return [i.strip() for i in issues_str.split(',') if i.strip()]


class ValidatorAgent(dspy.Module):
    """
    Enforces schema compliance and data typing
    
    CHECKS:
    - Schema Compliance: Structure matches defined schema?
    - Data Typing: Correct types for subject/predicate/object?
    - Constraint Enforcement: Domain/range constraints satisfied?
    """
    
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(ValidatorSignature)
    
    def forward(
        self,
        triple: Dict,
        ontology: Dict
    ) -> ValidationResult:
        """Validate schema compliance"""
        
        schema_str = self._format_schema(ontology)
        triple_str = f"({triple['subject']}, {triple['predicate']}, {triple['object']})"
        
        result = self.validate(
            ontology_schema=schema_str,
            triple=triple_str
        )
        
        # Calculate score
        checks = [
            result.schema_compliant,
            result.correct_data_types,
            result.constraints_satisfied
        ]
        score = sum(checks) / len(checks)
        
        issues = self._parse_issues(result.issues)
        
        return ValidationResult(
            passed=score >= 0.9,  # Strict 90% threshold for schema
            score=score,
            issues=issues,
            metadata={
                'schema_compliant': result.schema_compliant,
                'correct_types': result.correct_data_types,
                'constraints_met': result.constraints_satisfied
            }
        )
    
    def _format_schema(self, ontology: Dict) -> str:
        """Format schema with types and constraints"""
        parts = []
        
        # Add property constraints
        if 'object_properties' in ontology:
            for prop, details in ontology['object_properties'].items():
                domain = details.get('rdfs:domain', 'Entity')
                range_val = details.get('rdfs:range', 'Entity')
                parts.append(f"{prop}: {domain} -> {range_val}")
        
        return ' | '.join(parts[:10])  # Limit to avoid token overflow
    
    def _parse_issues(self, issues_str: str) -> List[str]:
        """Parse issues"""
        if issues_str.lower() in ['none', 'n/a', '']:
            return []
        return [i.strip() for i in issues_str.split(',') if i.strip()]


class CriticAgent(dspy.Module):
    """
    Detects duplicates, ambiguity, contradictions
    
    CHECKS:
    - Duplication: Is this a duplicate entity/relationship?
    - Relevance: Is this noise or meaningful?
    - Ambiguity: Are entities clearly identified?
    - Contradiction: Does this conflict with existing facts?
    """
    
    def __init__(self):
        super().__init__()
        self.critique = dspy.ChainOfThought(CriticSignature)
    
    def forward(
        self,
        triple: Dict,
        existing_triples: List[Dict],
        source_text: str
    ) -> ValidationResult:
        """Critique triple quality"""
        
        triple_str = f"({triple['subject']}, {triple['predicate']}, {triple['object']})"
        existing_str = self._format_existing_triples(existing_triples)
        
        result = self.critique(
            triple=triple_str,
            existing_triples=existing_str,
            source_text=source_text
        )
        
        # Calculate score (higher is better)
        checks = [
            not result.is_duplicate,  # Not duplicate is good
            result.is_relevant,
            result.is_unambiguous,
            result.no_contradictions
        ]
        score = sum(checks) / len(checks)
        
        issues = self._parse_issues(result.issues)
        
        return ValidationResult(
            passed=score >= 0.75,  # 75% threshold for critic
            score=score,
            issues=issues,
            metadata={
                'duplicate': result.is_duplicate,
                'relevant': result.is_relevant,
                'unambiguous': result.is_unambiguous,
                'no_contradictions': result.no_contradictions
            }
        )
    
    def _format_existing_triples(self, triples: List[Dict]) -> str:
        """Format existing triples for comparison"""
        if not triples:
            return "No existing triples"
        
        # Show last 5 triples to avoid token overflow
        recent = triples[-5:]
        formatted = [
            f"({t['subject']}, {t['predicate']}, {t['object']})"
            for t in recent
        ]
        return ' | '.join(formatted)
    
    def _parse_issues(self, issues_str: str) -> List[str]:
        """Parse issues"""
        if issues_str.lower() in ['none', 'n/a', '']:
            return []
        return [i.strip() for i in issues_str.split(',') if i.strip()]


# ================== Quality Control Pipeline ==================

class QualityControlPipeline:
    """
    Orchestrates Proposer -> Validator -> Critic workflow
    
    ACCEPTANCE CRITERIA:
    - Must pass all three agents with threshold scores
    - Proposer: 80% (ontology alignment, completeness)
    - Validator: 90% (schema compliance)
    - Critic: 75% (no duplicates/contradictions)
    - Overall: 85% combined score
    """
    
    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        acceptance_threshold: float = 0.85
    ):
        """
        Args:
            lm: DSPy language model (None = use default)
            acceptance_threshold: Minimum overall score to accept
        """
        if lm:
            dspy.settings.configure(lm=lm)
        
        self.acceptance_threshold = acceptance_threshold
        
        # Initialize agents
        self.proposer = ProposerAgent()
        self.validator = ValidatorAgent()
        self.critic = CriticAgent()
        
        # Track validated triples for critic
        self.validated_triples = []
    
    def validate_triple(
        self,
        triple: Dict,
        ontology: Dict,
        source_text: str
    ) -> TripleQuality:
        """
        Run triple through Proposer -> Validator -> Critic
        
        Returns quality assessment with pass/fail and reasons
        """
        quality = TripleQuality(triple=triple)
        
        # Stage 1: Proposer
        quality.proposer_result = self.proposer(
            triple=triple,
            ontology=ontology,
            source_text=source_text
        )
        
        if not quality.proposer_result.passed:
            # Early rejection - don't proceed
            return quality
        
        # Stage 2: Validator
        quality.validator_result = self.validator(
            triple=triple,
            ontology=ontology
        )
        
        if not quality.validator_result.passed:
            # Schema violation - reject
            return quality
        
        # Stage 3: Critic
        quality.critic_result = self.critic(
            triple=triple,
            existing_triples=self.validated_triples,
            source_text=source_text
        )
        
        # Check overall score
        if quality.passed and quality.overall_score >= self.acceptance_threshold:
            # Triple passed - add to validated set
            self.validated_triples.append(triple)
        
        return quality
    
    def validate_batch(
        self,
        triples: List[Dict],
        ontology: Dict,
        source_text: str
    ) -> Tuple[List[Dict], List[TripleQuality]]:
        """
        Validate a batch of triples
        
        Returns:
            (accepted_triples, all_quality_assessments)
        """
        accepted = []
        assessments = []
        
        for triple in triples:
            quality = self.validate_triple(triple, ontology, source_text)
            assessments.append(quality)
            
            if quality.passed:
                accepted.append(triple)
        
        return accepted, assessments
    
    def get_rejection_report(self, assessment: TripleQuality) -> str:
        """Generate human-readable rejection report"""
        if assessment.passed:
            return "✓ Triple accepted"
        
        report = [
            "✗ Triple rejected",
            f"  Overall score: {assessment.overall_score:.2%}",
            f"  Required: {self.acceptance_threshold:.2%}",
            "",
            "Issues:"
        ]
        
        # Group issues by agent
        if assessment.proposer_result and not assessment.proposer_result.passed:
            report.append(f"  Proposer ({assessment.proposer_result.score:.2%}):")
            for issue in assessment.proposer_result.issues:
                report.append(f"    - {issue}")
        
        if assessment.validator_result and not assessment.validator_result.passed:
            report.append(f"  Validator ({assessment.validator_result.score:.2%}):")
            for issue in assessment.validator_result.issues:
                report.append(f"    - {issue}")
        
        if assessment.critic_result and not assessment.critic_result.passed:
            report.append(f"  Critic ({assessment.critic_result.score:.2%}):")
            for issue in assessment.critic_result.issues:
                report.append(f"    - {issue}")
        
        return '\n'.join(report)
    
    def reset_context(self):
        """Clear validated triples (for new document)"""
        self.validated_triples = []


# ================== Configuration Helper ==================

def setup_dspy_for_context_graph(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> dspy.LM:
    """
    Configure DSPy with LLM provider
    
    Args:
        provider: 'openai' or 'gemini'
        model: Model name
        api_key: API key (None = use env var)
    
    Returns:
        Configured LM instance
    """
    if provider == "openai":
        lm = dspy.OpenAI(model=model, api_key=api_key)
    elif provider == "gemini":
        # DSPy may not have native Gemini support yet
        # Fallback to OpenAI-compatible endpoint or custom adapter
        raise NotImplementedError(
            "DSPy native Gemini support pending. "
            "Use OpenAI provider or implement custom adapter."
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    dspy.settings.configure(lm=lm)
    return lm
