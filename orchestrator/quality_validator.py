#!/usr/bin/env python3
"""
Multi-Agent Quality Validator
Validates output quality from CLI agents using multiple criteria
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    score: float
    passed: bool
    criteria_results: Dict[str, bool]
    feedback: List[str]

class MultiAgentQualityValidator:
    """Validates output quality from delegated agents"""
    
    def __init__(self):
        # Quality criteria for different output types
        self.validation_rules = {
            'code': {
                'min_length': 10,
                'has_syntax': True,
                'has_comments': False,  # Optional
                'executable': True,
                'follows_conventions': True
            },
            'research': {
                'min_length': 50,
                'has_structure': True,
                'has_sources': False,  # Optional
                'comprehensive': True,
                'accurate': True
            },
            'documentation': {
                'min_length': 30,
                'has_examples': False,  # Optional
                'well_formatted': True,
                'clear_language': True,
                'complete': True
            }
        }
    
    def is_healthy(self) -> bool:
        """Health check for quality validator"""
        return True
    
    async def validate_output(self, output: str, criteria: Dict[str, Any], task_type) -> float:
        """Validate output quality and return score"""
        if not output or not output.strip():
            return 0.0
        
        # Determine validation approach based on task type
        task_str = str(task_type).lower()
        
        if 'code' in task_str:
            return await self._validate_code_output(output, criteria)
        elif 'research' in task_str or 'analysis' in task_str:
            return await self._validate_research_output(output, criteria)
        elif 'documentation' in task_str:
            return await self._validate_documentation_output(output, criteria)
        else:
            return await self._validate_general_output(output, criteria)
    
    async def _validate_code_output(self, output: str, criteria: Dict[str, Any]) -> float:
        """Validate code output specifically"""
        score = 0.0
        max_score = 5.0
        
        # Length check
        if len(output.strip()) >= criteria.get('min_length', 10):
            score += 1.0
        
        # Syntax check (basic)
        if self._has_code_syntax(output):
            score += 1.5
        
        # Structure check
        if self._has_code_structure(output):
            score += 1.0
        
        # Comments check (bonus)
        if self._has_comments(output):
            score += 0.5
        
        # Executable check (basic pattern matching)
        if self._appears_executable(output):
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    async def _validate_research_output(self, output: str, criteria: Dict[str, Any]) -> float:
        """Validate research/analysis output"""
        score = 0.0
        max_score = 5.0
        
        # Length check
        if len(output.strip()) >= criteria.get('min_length', 50):
            score += 1.0
        
        # Structure check
        if self._has_clear_structure(output):
            score += 1.5
        
        # Depth check
        if self._has_analytical_depth(output):
            score += 1.5
        
        # Completeness check
        if self._appears_comprehensive(output):
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    async def _validate_documentation_output(self, output: str, criteria: Dict[str, Any]) -> float:
        """Validate documentation output"""
        score = 0.0
        max_score = 5.0
        
        # Length check
        if len(output.strip()) >= criteria.get('min_length', 30):
            score += 1.0
        
        # Formatting check
        if self._well_formatted(output):
            score += 1.5
        
        # Clarity check
        if self._has_clear_language(output):
            score += 1.0
        
        # Examples check (bonus)
        if self._has_examples(output):
            score += 0.5
        
        # Completeness check
        if self._appears_complete(output):
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    async def _validate_general_output(self, output: str, criteria: Dict[str, Any]) -> float:
        """Validate general output"""
        score = 0.0
        max_score = 3.0
        
        # Basic length check
        min_length = criteria.get('min_length', 10)
        if len(output.strip()) >= min_length:
            score += 1.0
        
        # Coherence check
        if self._is_coherent(output):
            score += 1.0
        
        # Quality threshold check
        quality_threshold = criteria.get('quality_threshold', 0.5)
        if len(output.split()) >= 5:  # Has substance
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    # Helper methods for validation
    def _has_code_syntax(self, output: str) -> bool:
        """Check if output contains code syntax"""
        code_indicators = [
            r'def\s+\w+\(',           # Python functions
            r'function\s+\w+\(',      # JavaScript functions
            r'class\s+\w+',           # Class definitions
            r'import\s+\w+',          # Import statements
            r'#include\s*<',          # C/C++ includes
            r'```\w*\n',              # Code blocks
            r'{\s*\n.*\n\s*}',        # Braces
            r'if\s*\(.*\)\s*{',       # If statements
            r'for\s*\(.*\)\s*{',      # For loops
            r'while\s*\(.*\)\s*{'     # While loops
        ]
        
        return any(re.search(pattern, output, re.MULTILINE | re.DOTALL) 
                  for pattern in code_indicators)
    
    def _has_code_structure(self, output: str) -> bool:
        """Check if code has proper structure"""
        structure_indicators = [
            r'def\s+\w+.*:',          # Function definitions
            r'class\s+\w+.*:',        # Class definitions
            r'try:\s*\n.*except',     # Error handling
            r'if\s+.*:\s*\n',         # Conditional logic
            r'return\s+',             # Return statements
            r'print\(',               # Output statements
            r'{\s*\n[\s\S]*\n\s*}'    # Block structures
        ]
        
        return any(re.search(pattern, output, re.MULTILINE | re.DOTALL)
                  for pattern in structure_indicators)
    
    def _has_comments(self, output: str) -> bool:
        """Check if code has comments"""
        comment_patterns = [
            r'#[^\n]*',               # Python comments
            r'//[^\n]*',              # C++/Java comments
            r'/\*[\s\S]*?\*/',        # Block comments
            r'"""[\s\S]*?"""',        # Python docstrings
            r"'''[\s\S]*?'''"         # Python docstrings
        ]
        
        return any(re.search(pattern, output) for pattern in comment_patterns)
    
    def _appears_executable(self, output: str) -> bool:
        """Check if code appears to be executable"""
        # Basic checks for executable code
        has_main_logic = any(pattern in output.lower() for pattern in [
            'def ', 'function', 'class', 'if ', 'for ', 'while ', 'return'
        ])
        
        has_syntax_errors = any(error in output for error in [
            'SyntaxError', 'IndentationError', 'NameError', 'undefined'
        ])
        
        return has_main_logic and not has_syntax_errors
    
    def _has_clear_structure(self, output: str) -> bool:
        """Check if text has clear structure"""
        structure_indicators = [
            r'^#+\s+',                # Headers
            r'^\d+\.\s+',             # Numbered lists
            r'^\*\s+',                # Bullet points
            r'^\-\s+',                # Dash lists
            r'\n\n',                  # Paragraph breaks
            r':\s*\n'                 # Colons for sections
        ]
        
        return any(re.search(pattern, output, re.MULTILINE)
                  for pattern in structure_indicators)
    
    def _has_analytical_depth(self, output: str) -> bool:
        """Check if research has analytical depth"""
        analysis_indicators = [
            'because', 'therefore', 'however', 'furthermore', 'moreover',
            'consequently', 'analysis', 'evaluation', 'comparison', 'conclusion',
            'evidence', 'indicates', 'suggests', 'demonstrates', 'reveals',
            'significant', 'important', 'impact', 'implications', 'factors'
        ]
        
        output_lower = output.lower()
        depth_score = sum(1 for indicator in analysis_indicators 
                         if indicator in output_lower)
        
        return depth_score >= 3  # At least 3 analytical terms
    
    def _appears_comprehensive(self, output: str) -> bool:
        """Check if output appears comprehensive"""
        # Check for multiple sections or substantial length
        sections = len(re.findall(r'\n\s*\n', output))  # Paragraph breaks
        words = len(output.split())
        
        return sections >= 2 and words >= 100
    
    def _well_formatted(self, output: str) -> bool:
        """Check if output is well formatted"""
        formatting_indicators = [
            r'^#+\s+.+$',             # Headers
            r'^\s*[\*\-\+]\s+',       # Lists
            r'^\s*\d+\.\s+',          # Numbered lists
            r'```[\s\S]*?```',        # Code blocks
            r'\*\*.*?\*\*',           # Bold text
            r'_.*?_',                 # Italic text
            r'\[.*?\]\(.*?\)'         # Links
        ]
        
        return any(re.search(pattern, output, re.MULTILINE)
                  for pattern in formatting_indicators)
    
    def _has_clear_language(self, output: str) -> bool:
        """Check if language is clear and professional"""
        # Check for clear, professional language
        clarity_indicators = [
            len(output.split()) >= 10,  # Substantial content
            not re.search(r'[.]{3,}', output),  # Not too many ellipses
            not re.search(r'[!]{2,}', output),  # Not too many exclamations
            len([s for s in output.split('.') if len(s.strip()) > 5]) >= 2  # Multiple sentences
        ]
        
        return sum(clarity_indicators) >= 3
    
    def _has_examples(self, output: str) -> bool:
        """Check if output contains examples"""
        example_indicators = [
            'example', 'for instance', 'such as', 'like', 'e.g.',
            '```', 'demo', 'sample', 'illustration', 'case'
        ]
        
        output_lower = output.lower()
        return any(indicator in output_lower for indicator in example_indicators)
    
    def _appears_complete(self, output: str) -> bool:
        """Check if output appears complete"""
        completeness_indicators = [
            len(output.split()) >= 50,  # Substantial length
            not output.endswith('...'),  # Not trailing off
            '.' in output[-10:],         # Proper ending
            not re.search(r'\[TODO\]|\[FIXME\]|\[PLACEHOLDER\]', output, re.IGNORECASE)
        ]
        
        return sum(completeness_indicators) >= 3
    
    def _is_coherent(self, output: str) -> bool:
        """Check if output is coherent"""
        # Basic coherence checks
        sentences = [s.strip() for s in output.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return len(output.split()) >= 5  # At least some content
        
        # Check for reasonable sentence lengths
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        return 3 <= avg_sentence_length <= 50  # Reasonable sentence length
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation system statistics"""
        return {
            'validation_rules': len(self.validation_rules),
            'supported_types': list(self.validation_rules.keys()),
            'total_criteria': sum(len(rules) for rules in self.validation_rules.values())
        }