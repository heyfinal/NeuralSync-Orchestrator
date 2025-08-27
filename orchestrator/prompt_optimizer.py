#!/usr/bin/env python3
"""
Bleeding-Edge Prompt Optimizer for CLI Agent Delegation
Creates optimally crafted prompts for each agent type to achieve maximum performance
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    CODEX_CODE_GENERATION = """You are an expert programmer. Generate clean, efficient, production-ready code.

Task: {prompt}

Requirements:
- Write complete, executable code
- Include error handling where appropriate
- Add brief inline comments for complex logic
- Follow best practices and conventions
- Optimize for readability and performance

Context: {context}

Code:"""

    CODEX_DEBUGGING = """You are a debugging expert. Analyze the code and fix all issues.

Code to debug:
{code}

Task: {prompt}

Requirements:
- Identify all bugs and issues
- Provide corrected code
- Explain what was wrong
- Add preventive measures

Context: {context}

Fixed code with explanations:"""

    GEMINI_RESEARCH = """You are a research analyst. Provide comprehensive, well-structured analysis.

Research Query: {prompt}

Instructions:
- Provide thorough, accurate information
- Include multiple perspectives when relevant
- Structure your response clearly with sections
- Cite reasoning and methodology
- Highlight key insights and implications

Context: {context}

Research Analysis:"""

    GEMINI_DOCUMENTATION = """You are a technical documentation expert. Create clear, comprehensive documentation.

Documentation Task: {prompt}

Requirements:
- Write clear, accessible explanations
- Include examples where helpful
- Structure with appropriate headings
- Cover edge cases and best practices
- Make it useful for different skill levels

Context: {context}

Documentation:"""

    CLAUDE_ORCHESTRATION = """You are an AI orchestration expert managing a team of specialized agents.

Orchestration Task: {prompt}

Your role:
- Analyze the request and break it into optimal subtasks
- Delegate to appropriate specialized agents (codex for code, gemini for research)
- Synthesize results into a cohesive final output
- Ensure quality and completeness

Available agents and their strengths:
- Codex: Code generation, debugging, testing, optimization
- Gemini: Research, analysis, documentation, planning

Context: {context}

Orchestration Plan:"""

class BleedingEdgePromptOptimizer:
    """Advanced prompt optimization engine for multi-agent delegation"""
    
    def __init__(self):
        self.agent_templates = {
            'codex': {
                'code_generation': PromptTemplate.CODEX_CODE_GENERATION,
                'debugging': PromptTemplate.CODEX_DEBUGGING,
                'default': PromptTemplate.CODEX_CODE_GENERATION
            },
            'gemini': {
                'research': PromptTemplate.GEMINI_RESEARCH,
                'analysis': PromptTemplate.GEMINI_RESEARCH,
                'documentation': PromptTemplate.GEMINI_DOCUMENTATION,
                'planning': PromptTemplate.GEMINI_DOCUMENTATION,
                'default': PromptTemplate.GEMINI_RESEARCH
            },
            'claude': {
                'orchestration': PromptTemplate.CLAUDE_ORCHESTRATION,
                'default': PromptTemplate.CLAUDE_ORCHESTRATION
            }
        }
        
        # Advanced prompt engineering techniques
        self.optimization_techniques = [
            'chain_of_thought',
            'few_shot_examples',
            'role_playing',
            'constraint_specification',
            'output_formatting',
            'context_injection'
        ]
        
    def is_healthy(self) -> bool:
        """Health check for prompt optimizer"""
        return True
    
    async def optimize_for_agent(self, prompt: str, agent: str, task_type, context: Dict[str, Any]) -> str:
        """Generate bleeding-edge optimized prompt for specific agent"""
        
        # Step 1: Select optimal template
        template = self._select_template(agent, task_type)
        
        # Step 2: Apply advanced optimization techniques
        optimized_prompt = await self._apply_optimizations(prompt, agent, task_type, context)
        
        # Step 3: Format with template
        formatted_prompt = self._format_with_template(template, optimized_prompt, context)
        
        # Step 4: Add agent-specific enhancements
        final_prompt = self._add_agent_enhancements(formatted_prompt, agent, task_type)
        
        logger.debug(f"Optimized prompt for {agent}: {len(final_prompt)} characters")
        return final_prompt
    
    def _select_template(self, agent: str, task_type) -> str:
        """Select optimal prompt template for agent and task"""
        agent_templates = self.agent_templates.get(agent, {})
        
        # Convert task_type to string key
        task_key = task_type.value if hasattr(task_type, 'value') else str(task_type)
        
        # Map task types to template keys
        task_mapping = {
            'code_generation': 'code_generation',
            'code_debugging': 'debugging', 
            'code_review': 'debugging',
            'testing': 'code_generation',
            'optimization': 'debugging',
            'research': 'research',
            'analysis': 'analysis',
            'documentation': 'documentation',
            'planning': 'planning',
            'orchestration': 'orchestration'
        }
        
        template_key = task_mapping.get(task_key, 'default')
        template = agent_templates.get(template_key, agent_templates.get('default'))
        
        return template.value if template else "{prompt}"
    
    async def _apply_optimizations(self, prompt: str, agent: str, task_type, context: Dict[str, Any]) -> str:
        """Apply advanced prompt optimization techniques"""
        optimized = prompt
        
        # Chain of Thought
        if agent in ['claude', 'gemini'] and self._is_complex_task(prompt):
            optimized = self._add_chain_of_thought(optimized)
        
        # Few-shot examples
        if agent == 'codex' and 'code' in str(task_type).lower():
            optimized = self._add_few_shot_examples(optimized, task_type)
        
        # Constraint specification
        optimized = self._add_constraints(optimized, agent, task_type)
        
        # Context integration
        optimized = self._integrate_context(optimized, context)
        
        return optimized
    
    def _is_complex_task(self, prompt: str) -> bool:
        """Determine if task is complex enough to benefit from chain-of-thought"""
        complexity_indicators = [
            len(prompt.split()) > 20,
            'analyze' in prompt.lower(),
            'compare' in prompt.lower(),
            'explain' in prompt.lower(),
            'multiple' in prompt.lower(),
            '?' in prompt  # Questions often benefit from reasoning
        ]
        return sum(complexity_indicators) >= 2
    
    def _add_chain_of_thought(self, prompt: str) -> str:
        """Add chain-of-thought reasoning prompts"""
        cot_addition = "\n\nThink through this step by step:\n1. First, understand the core requirements\n2. Consider the key challenges and constraints\n3. Develop a logical approach\n4. Provide your detailed response\n\nStep-by-step reasoning:"
        return prompt + cot_addition
    
    def _add_few_shot_examples(self, prompt: str, task_type) -> str:
        """Add relevant few-shot examples for coding tasks"""
        if 'generation' in str(task_type).lower():
            example = '''
Example of good code structure:
```python
def process_data(data: List[Dict]) -> Dict[str, Any]:
    """Process input data and return summary."""
    try:
        # Input validation
        if not data:
            return {"status": "empty", "count": 0}
        
        # Processing logic
        processed = [item for item in data if item.get("valid")]
        
        return {
            "status": "success",
            "count": len(processed),
            "data": processed
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {"status": "error", "error": str(e)}
```

Now for your task:'''
            return example + "\n\n" + prompt
        
        return prompt
    
    def _add_constraints(self, prompt: str, agent: str, task_type) -> str:
        """Add agent-specific constraints and requirements"""
        constraints = []
        
        if agent == 'codex':
            constraints.extend([
                "- Code must be production-ready and executable",
                "- Include appropriate error handling",
                "- Follow language-specific best practices",
                "- Add type hints where applicable"
            ])
        elif agent == 'gemini':
            constraints.extend([
                "- Provide comprehensive and accurate information",
                "- Structure response with clear sections",
                "- Include relevant examples or illustrations",
                "- Consider multiple perspectives where appropriate"
            ])
        elif agent == 'claude':
            constraints.extend([
                "- Break down complex tasks into manageable steps",
                "- Consider optimal resource allocation",
                "- Ensure quality control and validation",
                "- Provide clear reasoning for decisions"
            ])
        
        if constraints:
            constraint_text = "\n\nSpecific Requirements:\n" + "\n".join(constraints)
            return prompt + constraint_text
        
        return prompt
    
    def _integrate_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Intelligently integrate relevant context"""
        if not context:
            return prompt
        
        # Extract relevant context elements
        relevant_context = []
        
        # Previous outputs from other agents
        for key, value in context.items():
            if key.endswith('_output') and value:
                agent_name = key.replace('_output', '')
                relevant_context.append(f"Previous {agent_name} output: {str(value)[:200]}...")
        
        # Memory context
        if 'memory_items' in context and context['memory_items']:
            memory_summary = f"Relevant memory: {len(context['memory_items'])} related items"
            relevant_context.append(memory_summary)
        
        # User preferences or settings
        if 'preferences' in context:
            relevant_context.append(f"User preferences: {context['preferences']}")
        
        if relevant_context:
            context_text = "\n\nRelevant Context:\n" + "\n".join(relevant_context)
            return prompt + context_text
        
        return prompt
    
    def _format_with_template(self, template: str, prompt: str, context: Dict[str, Any]) -> str:
        """Format prompt with selected template"""
        # Extract code if present in context
        code = context.get('code', '')
        
        # Format context for template
        context_str = self._format_context_for_template(context)
        
        try:
            formatted = template.format(
                prompt=prompt,
                context=context_str,
                code=code
            )
            return formatted
        except KeyError as e:
            # Fallback if template formatting fails
            logger.warning(f"Template formatting failed: {e}")
            return f"{prompt}\n\nContext: {context_str}"
    
    def _format_context_for_template(self, context: Dict[str, Any]) -> str:
        """Format context dictionary for template inclusion"""
        if not context:
            return "No additional context provided."
        
        # Filter and format relevant context elements
        formatted_elements = []
        
        for key, value in context.items():
            if key.startswith('_') or not value:
                continue
                
            if isinstance(value, (list, dict)):
                formatted_elements.append(f"- {key}: {type(value).__name__} with {len(value)} items")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                formatted_elements.append(f"- {key}: {value_str}")
        
        return "\n".join(formatted_elements) if formatted_elements else "No specific context provided."
    
    def _add_agent_enhancements(self, prompt: str, agent: str, task_type) -> str:
        """Add final agent-specific enhancements"""
        
        if agent == 'codex':
            # Add execution context for code tasks
            enhancement = "\n\nPlease provide complete, runnable code with clear output."
            
        elif agent == 'gemini':
            # Add analytical depth request
            enhancement = "\n\nProvide thorough analysis with clear conclusions and actionable insights."
            
        elif agent == 'claude':
            # Add orchestration guidance
            enhancement = "\n\nConsider all aspects and provide comprehensive, well-reasoned responses."
            
        else:
            enhancement = ""
        
        return prompt + enhancement
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get prompt optimization statistics"""
        return {
            'available_templates': len(self.agent_templates),
            'optimization_techniques': len(self.optimization_techniques),
            'supported_agents': list(self.agent_templates.keys())
        }