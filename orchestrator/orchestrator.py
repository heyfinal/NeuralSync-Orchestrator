#!/usr/bin/env python3
"""
Claude Code Advanced Orchestration System
Transforms Claude Code into a master orchestrator that delegates tasks to codex and gemini
as extensions of itself, optimizing API usage while maintaining bleeding-edge quality.
"""

import asyncio
import json
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import re

from .prompt_optimizer import BleedingEdgePromptOptimizer
from .agent_monitor import AgentHealthMonitor
from .task_router import IntelligentTaskRouter
from .quality_validator import MultiAgentQualityValidator
from .delegation_memory import get_delegation_memory

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    ORCHESTRATION = "orchestration"
    TESTING = "testing"
    OPTIMIZATION = "optimization"

class AgentCapability(Enum):
    CODEX = {
        "strengths": ["code_generation", "debugging", "testing", "optimization"],
        "optimal_for": ["python", "javascript", "bash", "sql", "regex"],
        "response_style": "direct_executable",
        "max_context": 8000
    }
    GEMINI = {
        "strengths": ["research", "analysis", "documentation", "planning"],
        "optimal_for": ["explanations", "comparisons", "summaries", "insights"],
        "response_style": "comprehensive_analysis",
        "max_context": 32000
    }
    CLAUDE = {
        "strengths": ["orchestration", "complex_reasoning", "multi_step_tasks"],
        "optimal_for": ["delegation", "quality_control", "user_interaction"],
        "response_style": "conversational_detailed",
        "max_context": 200000
    }

@dataclass
class TaskAssignment:
    task_id: str
    agent: str
    task_type: TaskType
    priority: int
    prompt: str
    context: Dict[str, Any]
    estimated_duration: float
    dependencies: List[str]
    validation_criteria: Dict[str, Any]

@dataclass
class ExecutionResult:
    task_id: str
    agent: str
    success: bool
    output: str
    execution_time: float
    quality_score: float
    error_message: Optional[str] = None
    retry_count: int = 0

class ClaudeOrchestrator:
    """Advanced orchestration system for Claude Code delegation"""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / '.neuralsync'
        
        # Initialize core components
        self.prompt_optimizer = BleedingEdgePromptOptimizer()
        self.agent_monitor = AgentHealthMonitor()
        self.task_router = IntelligentTaskRouter()
        self.quality_validator = MultiAgentQualityValidator()
        self.delegation_memory = get_delegation_memory()
        
        # Execution tracking
        self.active_tasks: Dict[str, TaskAssignment] = {}
        self.execution_history: List[ExecutionResult] = []
        self.performance_metrics = {
            'claude_usage_reduction': 0.0,
            'avg_task_quality': 0.0,
            'delegation_success_rate': 0.0,
            'total_tasks_delegated': 0
        }
        
        # Agent availability
        self.available_agents = self._detect_available_agents()
        
    def _detect_available_agents(self) -> Dict[str, bool]:
        """Detect which CLI agents are available and functional"""
        agents = {}
        
        # Check codex-ns
        try:
            result = subprocess.run(['codex-ns', '--version'], 
                                  capture_output=True, timeout=5)
            agents['codex'] = result.returncode == 0
        except:
            agents['codex'] = False
            
        # Check gemini-ns
        try:
            result = subprocess.run(['gemini-ns', '--version'], 
                                  capture_output=True, timeout=5)
            agents['gemini'] = result.returncode == 0
        except:
            agents['gemini'] = False
            
        # Claude Code is always available (we're running in it)
        agents['claude'] = True
        
        logger.info(f"Available agents: {agents}")
        return agents
    
    async def orchestrate_task(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main orchestration method - analyzes request and delegates optimally
        """
        context = context or {}
        start_time = time.time()
        
        # Step 1: Analyze and route the task
        task_plan = await self.task_router.analyze_and_route(user_request, context)
        
        # Step 2: Optimize task plan using delegation memory
        optimized_plan = await self._optimize_with_memory(task_plan, context)
        
        # Step 3: Execute optimized task plan
        results = await self._execute_task_plan(optimized_plan, user_request)
        
        # Step 4: Validate and synthesize results
        final_result = await self._synthesize_results(results, user_request)
        
        # Step 5: Record delegation pattern for learning
        await self._record_delegation_pattern(task_plan, results, start_time, context)
        
        # Step 6: Update performance metrics
        self._update_performance_metrics(results)
        
        return final_result
    
    async def _execute_task_plan(self, task_plan: Dict[str, Any], original_request: str) -> List[ExecutionResult]:
        """Execute the planned tasks across appropriate agents"""
        results = []
        
        # Handle different execution strategies
        if task_plan['strategy'] == 'parallel':
            results = await self._execute_parallel_tasks(task_plan['tasks'])
        elif task_plan['strategy'] == 'sequential':
            results = await self._execute_sequential_tasks(task_plan['tasks'])
        else:  # hybrid
            results = await self._execute_hybrid_tasks(task_plan['tasks'])
            
        return results
    
    async def _execute_parallel_tasks(self, tasks: List[TaskAssignment]) -> List[ExecutionResult]:
        """Execute tasks in parallel for maximum efficiency"""
        semaphore = asyncio.Semaphore(3)  # Limit concurrent executions
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_single_task(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    task_id=tasks[i].task_id,
                    agent=tasks[i].agent,
                    success=False,
                    output="",
                    execution_time=0.0,
                    quality_score=0.0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_sequential_tasks(self, tasks: List[TaskAssignment]) -> List[ExecutionResult]:
        """Execute tasks sequentially, passing context between them"""
        results = []
        accumulated_context = {}
        
        for task in tasks:
            # Add previous results to context
            task.context.update(accumulated_context)
            
            # Execute task
            result = await self._execute_single_task(task)
            results.append(result)
            
            # Update accumulated context
            if result.success:
                accumulated_context[f"{task.agent}_output"] = result.output
                
        return results
    
    async def _execute_hybrid_tasks(self, tasks: List[TaskAssignment]) -> List[ExecutionResult]:
        """Execute tasks using hybrid strategy - parallel where possible, sequential where needed"""
        # Group tasks by dependencies
        independent_tasks = [t for t in tasks if not t.dependencies]
        dependent_tasks = [t for t in tasks if t.dependencies]
        
        results = []
        
        # Execute independent tasks in parallel first
        if independent_tasks:
            parallel_results = await self._execute_parallel_tasks(independent_tasks)
            results.extend(parallel_results)
        
        # Execute dependent tasks sequentially
        if dependent_tasks:
            sequential_results = await self._execute_sequential_tasks(dependent_tasks)
            results.extend(sequential_results)
            
        return results
    
    async def _execute_single_task(self, task: TaskAssignment) -> ExecutionResult:
        """Execute a single task on the assigned agent"""
        start_time = time.time()
        
        try:
            # Monitor agent health before execution
            if not await self.agent_monitor.is_agent_healthy(task.agent):
                logger.warning(f"Agent {task.agent} unhealthy, attempting recovery")
                await self.agent_monitor.recover_agent(task.agent)
            
            # Generate optimized prompt
            optimized_prompt = await self.prompt_optimizer.optimize_for_agent(
                task.prompt, task.agent, task.task_type, task.context
            )
            
            # Execute on the agent
            output = await self._call_agent(task.agent, optimized_prompt)
            
            # Validate output quality
            quality_score = await self.quality_validator.validate_output(
                output, task.validation_criteria, task.task_type
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                task_id=task.task_id,
                agent=task.agent,
                success=True,
                output=output,
                execution_time=execution_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed on {task.agent}: {e}")
            
            return ExecutionResult(
                task_id=task.task_id,
                agent=task.agent,
                success=False,
                output="",
                execution_time=execution_time,
                quality_score=0.0,
                error_message=str(e)
            )
    
    async def _call_agent(self, agent: str, prompt: str) -> str:
        """Make optimized calls to CLI agents"""
        if agent == 'codex':
            return await self._call_codex(prompt)
        elif agent == 'gemini':
            return await self._call_gemini(prompt)
        else:
            # Handle internal Claude processing
            return prompt  # Placeholder for Claude internal processing
    
    async def _call_codex(self, prompt: str) -> str:
        """Optimized codex-ns execution"""
        try:
            # Use exec mode for non-interactive execution
            process = await asyncio.create_subprocess_exec(
                'codex-ns', 'exec', '--skip-git-repo-check', prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            
            if process.returncode == 0:
                return stdout.decode('utf-8').strip()
            else:
                raise Exception(f"Codex execution failed: {stderr.decode('utf-8')}")
                
        except asyncio.TimeoutError:
            raise Exception("Codex execution timed out")
    
    async def _call_gemini(self, prompt: str) -> str:
        """Optimized gemini-ns execution"""
        try:
            # Use appropriate gemini flags for non-interactive execution
            process = await asyncio.create_subprocess_exec(
                'gemini-ns', prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            
            if process.returncode == 0:
                return stdout.decode('utf-8').strip()
            else:
                raise Exception(f"Gemini execution failed: {stderr.decode('utf-8')}")
                
        except asyncio.TimeoutError:
            raise Exception("Gemini execution timed out")
    
    async def _synthesize_results(self, results: List[ExecutionResult], original_request: str) -> Dict[str, Any]:
        """Synthesize results from multiple agents into a cohesive response"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Calculate overall metrics
        total_execution_time = sum(r.execution_time for r in results)
        avg_quality_score = sum(r.quality_score for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Synthesize outputs
        synthesized_output = await self._intelligent_synthesis(successful_results, original_request)
        
        return {
            'success': len(successful_results) > 0,
            'output': synthesized_output,
            'metrics': {
                'total_tasks': len(results),
                'successful_tasks': len(successful_results),
                'failed_tasks': len(failed_results),
                'total_execution_time': total_execution_time,
                'average_quality_score': avg_quality_score,
                'agents_used': list(set(r.agent for r in successful_results))
            },
            'detailed_results': [asdict(r) for r in results]
        }
    
    async def _intelligent_synthesis(self, results: List[ExecutionResult], original_request: str) -> str:
        """Intelligently combine outputs from multiple agents"""
        if not results:
            return "No successful results to synthesize."
        
        if len(results) == 1:
            return results[0].output
        
        # Multi-agent synthesis - combine complementary outputs
        synthesis_parts = []
        
        # Group by agent type for structured synthesis
        by_agent = {}
        for result in results:
            if result.agent not in by_agent:
                by_agent[result.agent] = []
            by_agent[result.agent].append(result.output)
        
        # Create structured synthesis
        for agent, outputs in by_agent.items():
            agent_synthesis = "\n".join(outputs)
            if agent == 'codex':
                synthesis_parts.append(f"## Code Implementation:\n{agent_synthesis}")
            elif agent == 'gemini':
                synthesis_parts.append(f"## Analysis & Insights:\n{agent_synthesis}")
            else:
                synthesis_parts.append(f"## {agent.title()} Output:\n{agent_synthesis}")
        
        return "\n\n".join(synthesis_parts)
    
    def _update_performance_metrics(self, results: List[ExecutionResult]):
        """Update orchestration performance metrics"""
        successful = len([r for r in results if r.success])
        total = len(results)
        
        if total > 0:
            # Update delegation success rate
            self.performance_metrics['delegation_success_rate'] = successful / total
            
            # Update average quality
            quality_scores = [r.quality_score for r in results if r.success]
            if quality_scores:
                self.performance_metrics['avg_task_quality'] = sum(quality_scores) / len(quality_scores)
            
            # Update usage reduction (estimate based on delegation)
            non_claude_tasks = len([r for r in results if r.agent != 'claude'])
            if total > 0:
                self.performance_metrics['claude_usage_reduction'] = non_claude_tasks / total
            
            # Update total delegated
            self.performance_metrics['total_tasks_delegated'] += total
    
    async def _optimize_with_memory(self, task_plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize task plan using learned delegation patterns"""
        
        # Extract task characteristics for memory lookup
        task_type = task_plan.get('primary_task_type', 'unknown')
        complexity = task_plan.get('complexity_score', 0.5)
        context_size = len(str(context))
        
        # Get optimal strategy from delegation memory
        memory_strategy = await self.delegation_memory.get_optimal_delegation_strategy(
            task_type, complexity, context_size
        )
        
        # Apply memory optimizations if confidence is high
        if memory_strategy.get('confidence', 0) > 0.6:
            logger.debug(f"Applying memory optimization: {memory_strategy}")
            
            # Override agents if memory suggests better ones
            if 'agents' in memory_strategy:
                available_memory_agents = [a for a in memory_strategy['agents'] 
                                         if self.available_agents.get(a, False)]
                if available_memory_agents:
                    # Update task assignments with memory-suggested agents
                    for task in task_plan.get('tasks', []):
                        if task.agent in available_memory_agents:
                            continue  # Keep existing good assignment
                        # Find best memory agent for this task
                        for memory_agent in available_memory_agents:
                            if self._agent_suitable_for_task(memory_agent, task.task_type):
                                task.agent = memory_agent
                                break
            
            # Override execution strategy if memory suggests better one
            if 'execution_strategy' in memory_strategy:
                task_plan['strategy'] = memory_strategy['execution_strategy']
            
            # Store memory optimization info for later recording
            task_plan['_memory_optimization'] = memory_strategy
        
        return task_plan
    
    def _agent_suitable_for_task(self, agent: str, task_type: str) -> bool:
        """Check if agent is suitable for task type"""
        agent_capabilities = {
            'codex': ['code_generation', 'code_debugging', 'testing', 'optimization'],
            'gemini': ['research', 'analysis', 'documentation', 'planning'],
            'claude': ['orchestration', 'synthesis']
        }
        
        return task_type in agent_capabilities.get(agent, [])
    
    async def _record_delegation_pattern(self, 
                                       task_plan: Dict[str, Any], 
                                       results: List[ExecutionResult], 
                                       start_time: float,
                                       context: Dict[str, Any]):
        """Record delegation pattern for learning"""
        
        if not results:
            return
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        success = len(successful_results) > 0
        
        # Calculate quality score (average of successful results)
        quality_score = 0.0
        if successful_results:
            quality_score = sum(r.quality_score for r in successful_results) / len(successful_results)
        
        # Get agents used
        agents_used = list(set(r.agent for r in results))
        
        # Estimate token usage (approximation)
        prompt_tokens = len(task_plan.get('prompt', '')) // 4  # Rough token estimation
        response_tokens = sum(len(r.output) // 4 for r in results)
        
        # Record the pattern
        await self.delegation_memory.record_delegation(
            task_type=task_plan.get('primary_task_type', 'unknown'),
            task_complexity=task_plan.get('complexity_score', 0.5),
            agents_used=agents_used,
            execution_time=execution_time,
            quality_score=quality_score,
            success=success,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            context_size=len(str(context)),
            execution_strategy=task_plan.get('strategy', 'sequential')
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics with delegation insights"""
        delegation_insights = self.delegation_memory.get_delegation_insights()
        
        return {
            'orchestration_metrics': self.performance_metrics,
            'available_agents': self.available_agents,
            'active_tasks': len(self.active_tasks),
            'execution_history_size': len(self.execution_history),
            'delegation_memory': {
                'patterns_learned': delegation_insights.get('total_patterns', 0),
                'success_rate': delegation_insights.get('success_rate', 0.0),
                'avg_quality': delegation_insights.get('avg_quality', 0.0),
                'top_agents': [agent.get('agent', 'unknown') for agent in delegation_insights.get('top_performing_agents', [])[:3]]
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the orchestration system"""
        health_status = {}
        
        # Check each agent
        for agent in ['codex', 'gemini', 'claude']:
            health_status[agent] = await self.agent_monitor.is_agent_healthy(agent)
        
        # Check system components
        health_status['components'] = {
            'prompt_optimizer': self.prompt_optimizer.is_healthy(),
            'task_router': self.task_router.is_healthy(),
            'quality_validator': self.quality_validator.is_healthy(),
            'delegation_memory': self.delegation_memory.is_healthy(),
            'agent_monitor': self.agent_monitor.is_healthy()
        }
        
        return health_status

# Global orchestrator instance
_orchestrator_instance: Optional[ClaudeOrchestrator] = None

def get_orchestrator() -> ClaudeOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ClaudeOrchestrator()
    return _orchestrator_instance

async def delegate_task(user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main delegation function for external use"""
    orchestrator = get_orchestrator()
    return await orchestrator.orchestrate_task(user_request, context)