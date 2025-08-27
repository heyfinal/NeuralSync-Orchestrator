#!/usr/bin/env python3
"""
Delegation Memory System for NeuralSync v2
Stores and learns from Claude Code orchestration patterns to optimize future delegations
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

@dataclass
class DelegationPattern:
    task_type: str
    task_complexity: float
    agents_used: List[str]
    execution_time: float
    quality_score: float
    success: bool
    prompt_tokens: int
    response_tokens: int
    created_at: float
    context_size: int
    execution_strategy: str  # parallel, sequential, hybrid

@dataclass
class AgentPerformance:
    agent: str
    total_tasks: int
    success_rate: float
    avg_response_time: float
    avg_quality_score: float
    preferred_task_types: List[str]
    token_efficiency: float
    last_updated: float

class DelegationMemorySystem:
    """Learns and optimizes Claude Code delegation patterns"""
    
    def __init__(self):
        self.patterns: List[DelegationPattern] = []
        self.agent_performance: Dict[str, AgentPerformance] = {}
        self.task_type_mappings: Dict[str, str] = {}  # task -> preferred agent
        self.optimization_cache: Dict[str, Dict] = {}  # cached optimizations
        self.memory_manager = get_memory_manager()
        
        # Learning parameters
        self.max_patterns = 10000  # Keep recent patterns for learning
        self.pattern_retention_days = 30
        self.learning_window = 100  # Patterns to analyze for optimization
        
        # Pattern analysis queues
        self.recent_patterns = deque(maxlen=self.learning_window)
        
        # Performance tracking
        self.delegation_stats = {
            'total_delegations': 0,
            'successful_delegations': 0,
            'avg_quality_improvement': 0.0,
            'token_savings': 0,
            'time_savings': 0.0
        }
        
        self._load_patterns()
    
    def is_healthy(self) -> bool:
        """Health check for delegation memory system"""
        return True
    
    async def record_delegation(self, 
                              task_type: str,
                              task_complexity: float,
                              agents_used: List[str],
                              execution_time: float,
                              quality_score: float,
                              success: bool,
                              prompt_tokens: int,
                              response_tokens: int,
                              context_size: int,
                              execution_strategy: str) -> None:
        """Record a delegation pattern for learning"""
        
        pattern = DelegationPattern(
            task_type=task_type,
            task_complexity=task_complexity,
            agents_used=agents_used,
            execution_time=execution_time,
            quality_score=quality_score,
            success=success,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            created_at=time.time(),
            context_size=context_size,
            execution_strategy=execution_strategy
        )
        
        # Add to memory
        self.patterns.append(pattern)
        self.recent_patterns.append(pattern)
        
        # Update agent performance
        for agent in agents_used:
            await self._update_agent_performance(agent, pattern)
        
        # Update delegation stats
        self.delegation_stats['total_delegations'] += 1
        if success:
            self.delegation_stats['successful_delegations'] += 1
        
        # Trigger learning if we have enough patterns
        if len(self.recent_patterns) >= 20:
            await self._analyze_recent_patterns()
        
        # Cleanup old patterns
        await self._cleanup_old_patterns()
    
    async def _update_agent_performance(self, agent: str, pattern: DelegationPattern):
        """Update performance metrics for an agent"""
        if agent not in self.agent_performance:
            self.agent_performance[agent] = AgentPerformance(
                agent=agent,
                total_tasks=0,
                success_rate=1.0,
                avg_response_time=pattern.execution_time,
                avg_quality_score=pattern.quality_score,
                preferred_task_types=[],
                token_efficiency=0.0,
                last_updated=time.time()
            )
        
        perf = self.agent_performance[agent]
        
        # Update rolling averages
        total_tasks = perf.total_tasks + 1
        success_rate = ((perf.success_rate * perf.total_tasks) + (1.0 if pattern.success else 0.0)) / total_tasks
        avg_response_time = ((perf.avg_response_time * perf.total_tasks) + pattern.execution_time) / total_tasks
        avg_quality_score = ((perf.avg_quality_score * perf.total_tasks) + pattern.quality_score) / total_tasks
        
        # Calculate token efficiency (quality per token)
        token_efficiency = avg_quality_score / max(1, (pattern.prompt_tokens + pattern.response_tokens) / 1000)
        
        # Update preferred task types
        if pattern.task_type not in perf.preferred_task_types and pattern.success and pattern.quality_score > 0.7:
            perf.preferred_task_types.append(pattern.task_type)
        
        # Update performance record
        perf.total_tasks = total_tasks
        perf.success_rate = success_rate
        perf.avg_response_time = avg_response_time
        perf.avg_quality_score = avg_quality_score
        perf.token_efficiency = token_efficiency
        perf.last_updated = time.time()
    
    async def get_optimal_delegation_strategy(self, 
                                            task_type: str, 
                                            task_complexity: float, 
                                            context_size: int) -> Dict[str, Any]:
        """Get optimal delegation strategy based on learned patterns"""
        
        # Check optimization cache first
        cache_key = f"{task_type}_{task_complexity:.2f}_{context_size}"
        if cache_key in self.optimization_cache:
            cached = self.optimization_cache[cache_key]
            # Use cache if less than 1 hour old
            if time.time() - cached['cached_at'] < 3600:
                return cached['strategy']
        
        # Analyze similar patterns
        similar_patterns = self._find_similar_patterns(task_type, task_complexity, context_size)
        
        if not similar_patterns:
            # Use default strategy
            strategy = self._get_default_strategy(task_type, task_complexity)
        else:
            # Optimize based on successful patterns
            strategy = self._optimize_from_patterns(similar_patterns, task_complexity)
        
        # Cache the optimization
        self.optimization_cache[cache_key] = {
            'strategy': strategy,
            'cached_at': time.time()
        }
        
        return strategy
    
    def _find_similar_patterns(self, task_type: str, complexity: float, context_size: int) -> List[DelegationPattern]:
        """Find patterns similar to current task"""
        similar = []
        
        for pattern in self.patterns[-500:]:  # Check recent patterns
            if not pattern.success:
                continue
                
            # Task type similarity
            type_similarity = 1.0 if pattern.task_type == task_type else 0.3
            
            # Complexity similarity (within 0.3 range)
            complexity_diff = abs(pattern.task_complexity - complexity)
            complexity_similarity = max(0, 1.0 - (complexity_diff / 0.3))
            
            # Context size similarity
            context_diff = abs(pattern.context_size - context_size) / max(pattern.context_size, context_size, 1)
            context_similarity = max(0, 1.0 - context_diff)
            
            # Overall similarity score
            similarity = (type_similarity * 0.5 + complexity_similarity * 0.3 + context_similarity * 0.2)
            
            if similarity > 0.6:  # Similarity threshold
                similar.append(pattern)
        
        # Sort by quality score and recency
        similar.sort(key=lambda p: (p.quality_score * 0.7 + (1.0 - (time.time() - p.created_at) / 86400) * 0.3), reverse=True)
        
        return similar[:20]  # Top 20 similar patterns
    
    def _get_default_strategy(self, task_type: str, complexity: float) -> Dict[str, Any]:
        """Get default delegation strategy for unknown patterns"""
        
        # Default agent mappings
        agent_mappings = {
            'code_generation': ['codex'],
            'code_debugging': ['codex'], 
            'code_review': ['codex'],
            'testing': ['codex'],
            'research': ['gemini'],
            'analysis': ['gemini'],
            'documentation': ['gemini'],
            'planning': ['claude'],
            'orchestration': ['claude']
        }
        
        # Default execution strategy
        execution_strategy = 'sequential' if complexity < 0.5 else 'parallel'
        
        return {
            'agents': agent_mappings.get(task_type, ['claude']),
            'execution_strategy': execution_strategy,
            'expected_quality': 0.75,
            'expected_time': 15.0,
            'confidence': 0.3  # Low confidence for defaults
        }
    
    def _optimize_from_patterns(self, patterns: List[DelegationPattern], complexity: float) -> Dict[str, Any]:
        """Optimize delegation strategy from similar successful patterns"""
        
        if not patterns:
            return self._get_default_strategy('unknown', complexity)
        
        # Analyze successful patterns
        best_patterns = [p for p in patterns if p.quality_score > 0.8][:10]
        
        if not best_patterns:
            best_patterns = patterns[:5]  # Use best available
        
        # Find most common successful agent combinations
        agent_combos = defaultdict(list)
        for pattern in best_patterns:
            agents_key = tuple(sorted(pattern.agents_used))
            agent_combos[agents_key].append(pattern)
        
        # Select best combination by average quality
        best_combo = None
        best_quality = 0
        
        for agents, combo_patterns in agent_combos.items():
            avg_quality = sum(p.quality_score for p in combo_patterns) / len(combo_patterns)
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_combo = agents
        
        # Determine execution strategy
        strategy_votes = defaultdict(int)
        for pattern in best_patterns:
            strategy_votes[pattern.execution_strategy] += 1
        
        execution_strategy = max(strategy_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate expected metrics
        expected_quality = sum(p.quality_score for p in best_patterns) / len(best_patterns)
        expected_time = sum(p.execution_time for p in best_patterns) / len(best_patterns)
        
        return {
            'agents': list(best_combo) if best_combo else ['claude'],
            'execution_strategy': execution_strategy,
            'expected_quality': expected_quality,
            'expected_time': expected_time,
            'confidence': min(0.9, len(patterns) / 20.0),  # Confidence based on pattern count
            'based_on_patterns': len(patterns)
        }
    
    async def _analyze_recent_patterns(self):
        """Analyze recent patterns for learning opportunities"""
        
        if len(self.recent_patterns) < 10:
            return
        
        # Find improvement opportunities
        improvements = self._identify_improvements()
        
        # Update task type mappings
        self._update_task_mappings()
        
        # Calculate performance improvements
        self._calculate_performance_gains()
        
        logger.debug(f"Analyzed {len(self.recent_patterns)} recent patterns, found {len(improvements)} improvements")
    
    def _identify_improvements(self) -> List[Dict]:
        """Identify opportunities to improve delegation patterns"""
        improvements = []
        
        # Group patterns by task type
        by_task_type = defaultdict(list)
        for pattern in self.recent_patterns:
            by_task_type[pattern.task_type].append(pattern)
        
        # Find underperforming task types
        for task_type, task_patterns in by_task_type.items():
            if len(task_patterns) < 3:
                continue
                
            avg_quality = sum(p.quality_score for p in task_patterns) / len(task_patterns)
            success_rate = sum(1 for p in task_patterns if p.success) / len(task_patterns)
            
            if avg_quality < 0.7 or success_rate < 0.8:
                improvements.append({
                    'task_type': task_type,
                    'issue': 'low_performance',
                    'avg_quality': avg_quality,
                    'success_rate': success_rate,
                    'recommendation': 'try_different_agents'
                })
        
        return improvements
    
    def _update_task_mappings(self):
        """Update task type to agent mappings based on performance"""
        
        # Analyze which agents perform best for each task type
        task_agent_performance = defaultdict(lambda: defaultdict(list))
        
        for pattern in self.patterns[-1000:]:  # Analyze recent 1000 patterns
            if not pattern.success:
                continue
                
            for agent in pattern.agents_used:
                task_agent_performance[pattern.task_type][agent].append(pattern.quality_score)
        
        # Update mappings for task types with enough data
        for task_type, agent_scores in task_agent_performance.items():
            if len(agent_scores) < 2:
                continue
                
            # Find best performing agent
            best_agent = None
            best_avg_quality = 0
            
            for agent, scores in agent_scores.items():
                if len(scores) >= 5:  # Need at least 5 samples
                    avg_quality = sum(scores) / len(scores)
                    if avg_quality > best_avg_quality:
                        best_avg_quality = avg_quality
                        best_agent = agent
            
            if best_agent and best_avg_quality > 0.75:
                self.task_type_mappings[task_type] = best_agent
    
    def _calculate_performance_gains(self):
        """Calculate performance improvements from delegation"""
        
        if len(self.patterns) < 50:
            return
            
        # Compare recent vs baseline performance
        recent_patterns = self.patterns[-50:]
        baseline_patterns = self.patterns[-200:-50] if len(self.patterns) >= 200 else []
        
        if not baseline_patterns:
            return
        
        # Calculate improvements
        recent_quality = sum(p.quality_score for p in recent_patterns if p.success) / max(1, sum(1 for p in recent_patterns if p.success))
        baseline_quality = sum(p.quality_score for p in baseline_patterns if p.success) / max(1, sum(1 for p in baseline_patterns if p.success))
        
        recent_time = sum(p.execution_time for p in recent_patterns) / len(recent_patterns)
        baseline_time = sum(p.execution_time for p in baseline_patterns) / len(baseline_patterns)
        
        # Update stats
        self.delegation_stats['avg_quality_improvement'] = recent_quality - baseline_quality
        self.delegation_stats['time_savings'] = baseline_time - recent_time
    
    async def _cleanup_old_patterns(self):
        """Remove old patterns to maintain memory limits"""
        
        current_time = time.time()
        cutoff_time = current_time - (self.pattern_retention_days * 86400)
        
        # Remove patterns older than retention period
        self.patterns = [p for p in self.patterns if p.created_at > cutoff_time]
        
        # Limit total pattern count
        if len(self.patterns) > self.max_patterns:
            # Keep most recent and highest quality patterns
            self.patterns.sort(key=lambda p: (p.created_at * 0.5 + p.quality_score * 0.5), reverse=True)
            self.patterns = self.patterns[:self.max_patterns]
    
    def get_delegation_insights(self) -> Dict[str, Any]:
        """Get insights about delegation patterns and performance"""
        
        if not self.patterns:
            return {'status': 'no_data', 'patterns': 0}
        
        recent_patterns = self.patterns[-100:] if len(self.patterns) >= 100 else self.patterns
        
        # Task type distribution
        task_types = defaultdict(int)
        for pattern in recent_patterns:
            task_types[pattern.task_type] += 1
        
        # Agent usage
        agent_usage = defaultdict(int)
        for pattern in recent_patterns:
            for agent in pattern.agents_used:
                agent_usage[agent] += 1
        
        # Performance metrics
        successful_patterns = [p for p in recent_patterns if p.success]
        
        insights = {
            'total_patterns': len(self.patterns),
            'recent_patterns': len(recent_patterns),
            'success_rate': len(successful_patterns) / len(recent_patterns) if recent_patterns else 0,
            'avg_quality': sum(p.quality_score for p in successful_patterns) / len(successful_patterns) if successful_patterns else 0,
            'avg_execution_time': sum(p.execution_time for p in recent_patterns) / len(recent_patterns) if recent_patterns else 0,
            'task_type_distribution': dict(task_types),
            'agent_usage': dict(agent_usage),
            'top_performing_agents': [
                {'agent': agent, **asdict(perf)} 
                for agent, perf in sorted(
                    self.agent_performance.items(),
                    key=lambda x: x[1].avg_quality_score * x[1].success_rate,
                    reverse=True
                )[:5]
            ],
            'delegation_stats': self.delegation_stats,
            'optimization_cache_size': len(self.optimization_cache)
        }
        
        return insights
    
    def _load_patterns(self):
        """Load stored patterns from disk"""
        try:
            patterns_file = Path.home() / '.neuralsync' / 'delegation_patterns.json'
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                
                # Load patterns
                for pattern_data in data.get('patterns', []):
                    pattern = DelegationPattern(**pattern_data)
                    self.patterns.append(pattern)
                    self.recent_patterns.append(pattern)
                
                # Load agent performance
                for agent, perf_data in data.get('agent_performance', {}).items():
                    self.agent_performance[agent] = AgentPerformance(**perf_data)
                
                # Load other data
                self.task_type_mappings = data.get('task_type_mappings', {})
                self.delegation_stats = {**self.delegation_stats, **data.get('delegation_stats', {})}
                
                logger.info(f"Loaded {len(self.patterns)} delegation patterns from storage")
                
        except Exception as e:
            logger.warning(f"Failed to load delegation patterns: {e}")
    
    async def save_patterns(self):
        """Save patterns to disk for persistence"""
        try:
            patterns_file = Path.home() / '.neuralsync' / 'delegation_patterns.json'
            patterns_file.parent.mkdir(exist_ok=True)
            
            data = {
                'patterns': [asdict(p) for p in self.patterns[-5000:]],  # Save recent patterns
                'agent_performance': {agent: asdict(perf) for agent, perf in self.agent_performance.items()},
                'task_type_mappings': self.task_type_mappings,
                'delegation_stats': self.delegation_stats,
                'saved_at': time.time()
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.patterns)} delegation patterns to storage")
            
        except Exception as e:
            logger.error(f"Failed to save delegation patterns: {e}")


# Global delegation memory instance
_global_delegation_memory: Optional[DelegationMemorySystem] = None

def get_delegation_memory() -> DelegationMemorySystem:
    """Get global delegation memory instance"""
    global _global_delegation_memory
    if _global_delegation_memory is None:
        _global_delegation_memory = DelegationMemorySystem()
    return _global_delegation_memory


async def analyze_delegation_effectiveness():
    """Analyze effectiveness of current delegation patterns"""
    memory = get_delegation_memory()
    insights = memory.get_delegation_insights()
    
    print("🧠 Delegation Memory Analysis")
    print("=" * 50)
    print(f"Total patterns learned: {insights['total_patterns']}")
    print(f"Success rate: {insights['success_rate']:.1%}")
    print(f"Average quality: {insights['avg_quality']:.2f}")
    print(f"Average execution time: {insights['avg_execution_time']:.1f}s")
    print(f"Top agents: {', '.join([a['agent'] for a in insights['top_performing_agents'][:3]])}")
    
    if insights['delegation_stats']['total_delegations'] > 0:
        print(f"Quality improvement: +{insights['delegation_stats']['avg_quality_improvement']:.3f}")
        print(f"Time savings: {insights['delegation_stats']['time_savings']:.1f}s")
    
    return insights


if __name__ == "__main__":
    asyncio.run(analyze_delegation_effectiveness())