"""
NeuralSync Orchestrator
Multi-agent delegation system for Claude
"""

from .orchestrator import ClaudeOrchestrator
from .task_router import IntelligentTaskRouter
from .prompt_optimizer import BleedingEdgePromptOptimizer
from .agent_monitor import AgentHealthMonitor
from .quality_validator import MultiAgentQualityValidator
from .delegation_memory import get_delegation_memory

__version__ = "1.0.0"
__author__ = "NeuralSync Team"

__all__ = [
    'ClaudeOrchestrator',
    'IntelligentTaskRouter', 
    'BleedingEdgePromptOptimizer',
    'AgentHealthMonitor',
    'MultiAgentQualityValidator',
    'get_delegation_memory'
]