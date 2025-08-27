#!/usr/bin/env python3
"""
Intelligent Task Router for Claude Code Orchestration
Analyzes user requests and optimally routes tasks to appropriate agents
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4

@dataclass 
class TaskRoute:
    agent: str
    confidence: float
    reasoning: str
    estimated_duration: float

class IntelligentTaskRouter:
    """Routes tasks to optimal agents based on content analysis"""
    
    def __init__(self):
        # Task pattern recognition
        self.code_patterns = [
            r'\b(write|create|generate|implement|build)\b.*\b(function|class|script|program|code)\b',
            r'\b(python|javascript|bash|sql|html|css|java|go|rust)\b',
            r'\b(debug|fix|error|bug|issue)\b',
            r'\btest\b.*\bcode\b',
            r'\boptimize\b.*\b(performance|speed|memory)\b'
        ]
        
        self.research_patterns = [
            r'\b(research|investigate|analyze|compare|study)\b',
            r'\b(what is|how does|explain|describe)\b',
            r'\b(pros and cons|advantages|disadvantages)\b',
            r'\b(summary|overview|report)\b',
            r'\b(find|search|lookup|discover)\b'
        ]
        
        self.planning_patterns = [
            r'\b(plan|strategy|approach|roadmap|architecture)\b',
            r'\b(design|structure|organize)\b',
            r'\b(requirements|specifications|documentation)\b'
        ]
        
        # Agent capabilities matrix
        self.agent_strengths = {
            'codex': {
                'primary': ['code_generation', 'debugging', 'testing', 'optimization'],
                'languages': ['python', 'javascript', 'bash', 'sql', 'html', 'css'],
                'max_context': 8000,
                'speed': 0.8,
                'code_quality': 0.9
            },
            'gemini': {
                'primary': ['research', 'analysis', 'documentation', 'planning'],
                'languages': ['natural_language', 'markdown', 'documentation'],
                'max_context': 32000,
                'speed': 0.7,
                'analysis_depth': 0.9
            },
            'claude': {
                'primary': ['orchestration', 'complex_reasoning', 'user_interaction'],
                'languages': ['natural_language', 'all_programming'],
                'max_context': 200000,
                'speed': 0.6,
                'reasoning_quality': 0.95
            }
        }
        
    def is_healthy(self) -> bool:
        """Health check for task router"""
        return True
        
    async def analyze_and_route(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze request and create optimal task routing plan"""
        context = context or {}
        
        # Step 1: Analyze request complexity and type
        analysis = self._analyze_request(user_request)
        
        # Step 2: Route to appropriate agents
        task_routes = self._route_tasks(analysis, context)
        
        # Step 3: Determine execution strategy
        execution_strategy = self._determine_strategy(task_routes, analysis)
        
        # Step 4: Create task assignments
        task_assignments = self._create_task_assignments(task_routes, user_request, context)
        
        return {
            'analysis': analysis,
            'routes': task_routes,
            'strategy': execution_strategy,
            'tasks': task_assignments,
            'estimated_total_time': sum(task.estimated_duration for task in task_assignments)
        }
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """Analyze the user request for content, complexity, and intent"""
        request_lower = request.lower()
        
        # Detect task types
        is_code_task = any(re.search(pattern, request_lower) for pattern in self.code_patterns)
        is_research_task = any(re.search(pattern, request_lower) for pattern in self.research_patterns)
        is_planning_task = any(re.search(pattern, request_lower) for pattern in self.planning_patterns)
        
        # Detect programming languages
        languages = []
        for lang in ['python', 'javascript', 'bash', 'sql', 'html', 'css', 'java', 'go', 'rust']:
            if lang in request_lower:
                languages.append(lang)
        
        # Estimate complexity
        complexity_indicators = [
            len(request.split()) > 50,  # Long requests
            'complex' in request_lower or 'advanced' in request_lower,
            'multiple' in request_lower or 'several' in request_lower,
            is_code_task and is_research_task,  # Hybrid tasks
            len(languages) > 2  # Multiple languages
        ]
        
        complexity_score = sum(complexity_indicators)
        if complexity_score >= 3:
            complexity = TaskComplexity.VERY_COMPLEX
        elif complexity_score == 2:
            complexity = TaskComplexity.COMPLEX
        elif complexity_score == 1:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE
        
        return {
            'request_length': len(request),
            'word_count': len(request.split()),
            'is_code_task': is_code_task,
            'is_research_task': is_research_task, 
            'is_planning_task': is_planning_task,
            'languages': languages,
            'complexity': complexity,
            'complexity_score': complexity_score,
            'requires_multi_agent': complexity_score >= 2
        }
    
    def _route_tasks(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[TaskRoute]:
        """Route tasks to optimal agents based on analysis"""
        routes = []
        
        # Primary routing logic
        if analysis['is_code_task']:
            # Code-related tasks go to codex primarily
            confidence = 0.9 if analysis['languages'] else 0.7
            routes.append(TaskRoute(
                agent='codex',
                confidence=confidence,
                reasoning=f"Code task detected with languages: {analysis['languages']}",
                estimated_duration=self._estimate_duration('codex', analysis)
            ))
            
            # For complex code tasks, also involve Claude for oversight
            if analysis['complexity'] in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                routes.append(TaskRoute(
                    agent='claude',
                    confidence=0.6,
                    reasoning="Complex code task requires oversight and quality control",
                    estimated_duration=self._estimate_duration('claude', analysis) * 0.3
                ))
        
        if analysis['is_research_task']:
            # Research tasks go to gemini
            routes.append(TaskRoute(
                agent='gemini',
                confidence=0.85,
                reasoning="Research/analysis task detected",
                estimated_duration=self._estimate_duration('gemini', analysis)
            ))
        
        if analysis['is_planning_task']:
            # Planning tasks can go to either gemini or claude
            if analysis['complexity'] in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                routes.append(TaskRoute(
                    agent='claude',
                    confidence=0.8,
                    reasoning="Complex planning requires advanced reasoning",
                    estimated_duration=self._estimate_duration('claude', analysis)
                ))
            else:
                routes.append(TaskRoute(
                    agent='gemini',
                    confidence=0.75,
                    reasoning="Planning task suitable for analysis engine",
                    estimated_duration=self._estimate_duration('gemini', analysis)
                ))
        
        # If no specific patterns matched, default routing
        if not routes:
            if analysis['complexity'] in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
                routes.append(TaskRoute(
                    agent='claude',
                    confidence=0.6,
                    reasoning="Complex general task requires advanced reasoning",
                    estimated_duration=self._estimate_duration('claude', analysis)
                ))
            else:
                # Route based on content hints
                if any(word in context.get('previous_agents', []) for word in ['codex', 'code']):
                    routes.append(TaskRoute(
                        agent='codex',
                        confidence=0.5,
                        reasoning="Previous context suggests code focus",
                        estimated_duration=self._estimate_duration('codex', analysis)
                    ))
                else:
                    routes.append(TaskRoute(
                        agent='gemini',
                        confidence=0.5,
                        reasoning="General task suitable for analysis",
                        estimated_duration=self._estimate_duration('gemini', analysis)
                    ))
        
        return routes
    
    def _determine_strategy(self, routes: List[TaskRoute], analysis: Dict[str, Any]) -> str:
        """Determine optimal execution strategy"""
        if len(routes) == 1:
            return 'single'
        
        # Check if tasks can be parallelized
        agents = [route.agent for route in routes]
        if len(set(agents)) == len(agents):  # All different agents
            if analysis['requires_multi_agent']:
                return 'parallel'
            else:
                return 'sequential'
        
        # Mixed strategy for complex tasks
        if analysis['complexity'] in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            return 'hybrid'
        
        return 'sequential'
    
    def _estimate_duration(self, agent: str, analysis: Dict[str, Any]) -> float:
        """Estimate task duration for an agent"""
        base_time = {
            'codex': 15.0,    # seconds
            'gemini': 20.0,   # seconds  
            'claude': 25.0    # seconds
        }
        
        # Adjust based on complexity
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 2.0,
            TaskComplexity.VERY_COMPLEX: 3.0
        }
        
        # Adjust based on request length
        length_multiplier = min(2.0, analysis['word_count'] / 100.0)
        
        duration = base_time[agent] * complexity_multiplier[analysis['complexity']] * length_multiplier
        return max(5.0, duration)  # Minimum 5 seconds
    
    def _create_task_assignments(self, routes: List[TaskRoute], user_request: str, context: Dict[str, Any]) -> List:
        """Create task assignment objects from routes"""
        from .claude_orchestrator import TaskAssignment, TaskType
        import uuid
        
        assignments = []
        
        for i, route in enumerate(routes):
            # Determine task type
            if route.agent == 'codex':
                task_type = TaskType.CODE_GENERATION
            elif route.agent == 'gemini':
                if 'research' in user_request.lower():
                    task_type = TaskType.RESEARCH
                elif 'plan' in user_request.lower():
                    task_type = TaskType.PLANNING
                else:
                    task_type = TaskType.ANALYSIS
            else:  # claude
                task_type = TaskType.ORCHESTRATION
            
            # Create validation criteria
            validation_criteria = {
                'min_length': 10,
                'max_length': 10000,
                'required_elements': [],
                'quality_threshold': 0.7
            }
            
            if route.agent == 'codex':
                validation_criteria['required_elements'] = ['executable', 'syntactically_correct']
            elif route.agent == 'gemini':
                validation_criteria['required_elements'] = ['informative', 'well_structured']
            
            assignment = TaskAssignment(
                task_id=str(uuid.uuid4()),
                agent=route.agent,
                task_type=task_type,
                priority=len(routes) - i,  # Earlier routes have higher priority
                prompt=user_request,  # Will be optimized later
                context=context,
                estimated_duration=route.estimated_duration,
                dependencies=[],  # Will be set based on strategy
                validation_criteria=validation_criteria
            )
            
            assignments.append(assignment)
        
        return assignments