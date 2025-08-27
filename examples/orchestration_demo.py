#!/usr/bin/env python3
"""
NeuralSync v2 Orchestration Demo
Example usage of the multi-agent delegation system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add NeuralSync to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralsync.claude_orchestrator import ClaudeOrchestrator
from neuralsync.delegation_memory import get_delegation_memory, analyze_delegation_effectiveness

async def demo_basic_orchestration():
    """Demo basic task orchestration"""
    print("🎭 Basic Orchestration Demo")
    print("=" * 40)
    
    orchestrator = ClaudeOrchestrator()
    
    # Example tasks for different agents
    tasks = [
        {
            'description': 'Code Generation (→ Codex)',
            'request': 'Write a Python function to calculate the factorial of a number with error handling'
        },
        {
            'description': 'Research Task (→ Gemini)', 
            'request': 'Research the key differences between REST and GraphQL APIs'
        },
        {
            'description': 'Code Review (→ Codex)',
            'request': 'Review this Python code for improvements: def calc(x): return x*x if x > 0 else None'
        }
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. {task['description']}")
        print(f"   Request: {task['request']}")
        
        try:
            result = await orchestrator.orchestrate_task(task['request'])
            
            print(f"   ✅ Success: {result.get('success', False)}")
            print(f"   🤖 Agents: {result.get('agents_used', 'Unknown')}")
            print(f"   ⏱️  Duration: {result.get('execution_time', 0):.1f}s")
            print(f"   📊 Quality: {result.get('quality_score', 0):.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("   " + "-" * 30)

async def demo_system_health():
    """Demo system health monitoring"""
    print("\n\n🏥 System Health Demo")
    print("=" * 40)
    
    orchestrator = ClaudeOrchestrator()
    
    # Check overall health
    health = await orchestrator.health_check()
    
    print("Agent Health:")
    for agent, status in health.items():
        if agent != 'components':
            icon = "✅" if status else "❌"
            print(f"  {icon} {agent}: {'Healthy' if status else 'Unavailable'}")
    
    print("\nComponent Health:")
    for component, status in health.get('components', {}).items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component}: {'OK' if status else 'Failed'}")
    
    # Get performance metrics
    metrics = orchestrator.get_performance_summary()
    print(f"\nPerformance Metrics:")
    print(f"  📈 Available agents: {list(metrics['available_agents'].keys())}")
    print(f"  📊 Active tasks: {metrics['active_tasks']}")
    
    if 'delegation_memory' in metrics:
        memory_info = metrics['delegation_memory']
        print(f"  🧠 Patterns learned: {memory_info['patterns_learned']}")
        print(f"  ✅ Success rate: {memory_info['success_rate']:.1%}")

async def demo_memory_learning():
    """Demo delegation memory and learning"""
    print("\n\n🧠 Memory Learning Demo")
    print("=" * 40)
    
    # Get delegation memory system
    memory = get_delegation_memory()
    
    # Run similar tasks to demonstrate learning
    orchestrator = ClaudeOrchestrator()
    
    similar_tasks = [
        'Create a Python function to validate email formats',
        'Write a Python function to check if a string is a valid URL',
        'Build a Python function to parse JSON safely'
    ]
    
    print("Running similar tasks to demonstrate learning...")
    
    for i, task in enumerate(similar_tasks, 1):
        print(f"\n  Task {i}: {task}")
        try:
            result = await orchestrator.orchestrate_task(task)
            print(f"    Result: {'✅' if result.get('success') else '❌'}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Show what the system learned
    print("\n📈 Memory Analysis:")
    await analyze_delegation_effectiveness()

async def demo_advanced_features():
    """Demo advanced orchestration features"""
    print("\n\n⚡ Advanced Features Demo") 
    print("=" * 40)
    
    orchestrator = ClaudeOrchestrator()
    
    # Complex multi-step task
    complex_task = """
    Create a complete Python web API that:
    1. Has a /health endpoint
    2. Uses FastAPI framework
    3. Includes error handling
    4. Has proper logging
    5. Includes basic documentation
    """
    
    print("🎪 Complex Multi-Step Task:")
    print(f"   {complex_task.strip()}")
    
    try:
        result = await orchestrator.orchestrate_task(complex_task)
        
        print(f"\n   ✅ Success: {result.get('success', False)}")
        print(f"   🤖 Agents used: {result.get('agents_used', [])}")
        print(f"   📊 Quality: {result.get('quality_score', 0):.2f}")
        print(f"   ⏱️  Time: {result.get('execution_time', 0):.1f}s")
        
        if result.get('final_output'):
            output_preview = result['final_output'][:200] + "..." if len(result['final_output']) > 200 else result['final_output']
            print(f"   📄 Output preview:\n{output_preview}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def demo_configuration():
    """Demo configuration options"""
    print("\n\n⚙️  Configuration Demo")
    print("=" * 40)
    
    config_file = Path.home() / '.neuralsync' / 'orchestration.json'
    
    if config_file.exists():
        print(f"✅ Configuration found: {config_file}")
        
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        print("📋 Current Settings:")
        print(f"   • Auto delegation: {config.get('delegation', {}).get('auto_delegate', False)}")
        print(f"   • Quality threshold: {config.get('delegation', {}).get('quality_threshold', 0.7)}")
        print(f"   • Codex enabled: {config.get('agents', {}).get('codex', {}).get('enabled', True)}")
        print(f"   • Gemini enabled: {config.get('agents', {}).get('gemini', {}).get('enabled', True)}")
        print(f"   • Memory learning: {config.get('memory', {}).get('learn_patterns', True)}")
    else:
        print("⚠️  No configuration file found")
        print(f"   Expected at: {config_file}")
        print("   Run: python3 install_orchestration.py")

async def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              NeuralSync v2 Orchestration Demo               ║
║                                                              ║
║  Demonstration of Claude as master orchestrator delegating  ║
║  tasks to codex and gemini as extensions of itself         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if orchestration is installed
    try:
        from neuralsync.claude_orchestrator import ClaudeOrchestrator
    except ImportError:
        print("❌ Orchestration module not installed!")
        print("   Run: python3 install_orchestration.py")
        return 1
    
    try:
        # Run demos
        await demo_basic_orchestration()
        await demo_system_health() 
        await demo_memory_learning()
        await demo_advanced_features()
        demo_configuration()
        
        print("\n" + "="*60)
        print("🎉 Orchestration Demo Complete!")
        print("   The system is learning and will improve over time.")
        print("   Check ~/.neuralsync/delegation_patterns.json for learned patterns.")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))