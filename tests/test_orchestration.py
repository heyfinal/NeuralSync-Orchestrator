#!/usr/bin/env python3
"""
Comprehensive Test Suite for NeuralSync v2 Multi-Agent Orchestration
Tests the complete Claude Code delegation system with memory learning
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add neuralsync to path
sys.path.insert(0, str(Path(__file__).parent))

from neuralsync.claude_orchestrator import ClaudeOrchestrator
from neuralsync.delegation_memory import get_delegation_memory, analyze_delegation_effectiveness

class OrchestrationTester:
    """Comprehensive test suite for the orchestration system"""
    
    def __init__(self):
        self.orchestrator = ClaudeOrchestrator()
        self.test_results = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🎯 NeuralSync v2 Orchestration Test Suite")
        print("=" * 60)
        
        # Test system health first
        await self.test_system_health()
        
        # Test basic orchestration capabilities
        await self.test_basic_delegation()
        
        # Test memory learning system
        await self.test_memory_learning()
        
        # Test complex multi-step tasks
        await self.test_complex_orchestration()
        
        # Test performance optimization
        await self.test_performance_optimization()
        
        # Generate final report
        await self.generate_test_report()
    
    async def test_system_health(self):
        """Test system health and component availability"""
        print("\n🏥 Testing System Health")
        print("-" * 30)
        
        try:
            health = await self.orchestrator.health_check()
            
            print(f"Agent Status:")
            for agent, status in health.items():
                if agent != 'components':
                    icon = "✅" if status else "❌"
                    print(f"  {icon} {agent}: {'Healthy' if status else 'Unavailable'}")
            
            print(f"\nComponent Status:")
            for component, status in health.get('components', {}).items():
                icon = "✅" if status else "❌"
                print(f"  {icon} {component}: {'OK' if status else 'Failed'}")
            
            # Test result
            all_healthy = all(health.get('components', {}).values())
            self.test_results.append({
                'test': 'system_health',
                'passed': all_healthy,
                'details': health
            })
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            self.test_results.append({
                'test': 'system_health',
                'passed': False,
                'error': str(e)
            })
    
    async def test_basic_delegation(self):
        """Test basic task delegation to codex and gemini"""
        print("\n🎭 Testing Basic Delegation")
        print("-" * 30)
        
        test_cases = [
            {
                'name': 'Code Generation',
                'request': 'Write a Python function to calculate fibonacci numbers',
                'expected_agent': 'codex'
            },
            {
                'name': 'Research Task',
                'request': 'Research the benefits of async programming in Python',
                'expected_agent': 'gemini'
            },
            {
                'name': 'Code Debugging',
                'request': 'Fix this Python code: def fib(n): return fib(n-1) + fib(n-2)',
                'expected_agent': 'codex'
            }
        ]
        
        for test_case in test_cases:
            print(f"\n  Testing: {test_case['name']}")
            
            try:
                start = time.time()
                result = await self.orchestrator.orchestrate_task(test_case['request'])
                duration = time.time() - start
                
                success = result.get('success', False)
                quality = result.get('quality_score', 0.0)
                
                print(f"    Result: {'✅' if success else '❌'} ({duration:.1f}s, Quality: {quality:.2f})")
                
                self.test_results.append({
                    'test': f'delegation_{test_case["name"].lower().replace(" ", "_")}',
                    'passed': success,
                    'duration': duration,
                    'quality': quality,
                    'details': result
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.test_results.append({
                    'test': f'delegation_{test_case["name"].lower().replace(" ", "_")}',
                    'passed': False,
                    'error': str(e)
                })
    
    async def test_memory_learning(self):
        """Test delegation memory learning system"""
        print("\n🧠 Testing Memory Learning System")
        print("-" * 30)
        
        # Get memory system
        memory = get_delegation_memory()
        
        # Test repeated similar tasks to see learning
        similar_tasks = [
            'Create a Python function to sort a list',
            'Write a Python function to reverse a string', 
            'Build a Python function to find max in array'
        ]
        
        print("  Running similar tasks to test learning...")
        
        for i, task in enumerate(similar_tasks):
            try:
                start = time.time()
                result = await self.orchestrator.orchestrate_task(task)
                duration = time.time() - start
                
                print(f"    Task {i+1}: {duration:.1f}s, Quality: {result.get('quality_score', 0):.2f}")
                
                # Small delay to allow memory processing
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Task {i+1} failed: {e}")
        
        # Test memory insights
        try:
            insights = memory.get_delegation_insights()
            
            print(f"\n  Memory Status:")
            print(f"    Patterns learned: {insights.get('total_patterns', 0)}")
            print(f"    Success rate: {insights.get('success_rate', 0):.1%}")
            print(f"    Avg quality: {insights.get('avg_quality', 0):.2f}")
            
            self.test_results.append({
                'test': 'memory_learning',
                'passed': insights.get('total_patterns', 0) > 0,
                'insights': insights
            })
            
        except Exception as e:
            print(f"    ❌ Memory analysis failed: {e}")
            self.test_results.append({
                'test': 'memory_learning',
                'passed': False,
                'error': str(e)
            })
    
    async def test_complex_orchestration(self):
        """Test complex multi-step orchestration"""
        print("\n🎪 Testing Complex Orchestration")
        print("-" * 30)
        
        complex_request = """
        Create a Python web scraper that:
        1. Scrapes product data from a website
        2. Stores the data in a database
        3. Includes error handling and logging
        4. Has proper documentation
        """
        
        print("  Running complex multi-step task...")
        
        try:
            start = time.time()
            result = await self.orchestrator.orchestrate_task(complex_request)
            duration = time.time() - start
            
            success = result.get('success', False)
            quality = result.get('quality_score', 0.0)
            
            print(f"    Result: {'✅' if success else '❌'}")
            print(f"    Duration: {duration:.1f}s")
            print(f"    Quality: {quality:.2f}")
            print(f"    Agents used: {result.get('agents_used', [])}")
            
            self.test_results.append({
                'test': 'complex_orchestration',
                'passed': success,
                'duration': duration,
                'quality': quality,
                'agents_used': result.get('agents_used', [])
            })
            
        except Exception as e:
            print(f"    ❌ Complex orchestration failed: {e}")
            self.test_results.append({
                'test': 'complex_orchestration',
                'passed': False,
                'error': str(e)
            })
    
    async def test_performance_optimization(self):
        """Test performance optimization features"""
        print("\n⚡ Testing Performance Optimization")
        print("-" * 30)
        
        # Run same task multiple times to test optimization
        test_task = "Write a Python function to calculate prime numbers"
        durations = []
        
        for i in range(3):
            try:
                start = time.time()
                result = await self.orchestrator.orchestrate_task(test_task)
                duration = time.time() - start
                durations.append(duration)
                
                print(f"    Run {i+1}: {duration:.1f}s (Quality: {result.get('quality_score', 0):.2f})")
                
            except Exception as e:
                print(f"    ❌ Run {i+1} failed: {e}")
                durations.append(None)
        
        # Check if performance improved (later runs should be faster due to caching/learning)
        valid_durations = [d for d in durations if d is not None]
        
        if len(valid_durations) >= 2:
            improvement = valid_durations[0] - valid_durations[-1]
            improved = improvement > 0
            
            print(f"    Performance improvement: {improvement:.1f}s ({'✅' if improved else '❌'})")
            
            self.test_results.append({
                'test': 'performance_optimization',
                'passed': improved,
                'improvement': improvement,
                'durations': durations
            })
        else:
            self.test_results.append({
                'test': 'performance_optimization',
                'passed': False,
                'error': 'Not enough successful runs'
            })
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Test Report")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get('passed', False)])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")
        print(f"Total Duration: {time.time() - self.start_time:.1f}s")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result.get('passed', False) else "❌"
            test_name = result['test'].replace('_', ' ').title()
            print(f"  {status} {test_name}")
            
            if result.get('duration'):
                print(f"      Duration: {result['duration']:.1f}s")
            if result.get('quality'):
                print(f"      Quality: {result['quality']:.2f}")
            if result.get('error'):
                print(f"      Error: {result['error']}")
        
        # System performance summary
        print(f"\nSystem Performance:")
        performance = self.orchestrator.get_performance_summary()
        
        print(f"  Available agents: {list(performance.get('available_agents', {}).keys())}")
        print(f"  Orchestration metrics: {performance.get('orchestration_metrics', {})}")
        
        if 'delegation_memory' in performance:
            memory_info = performance['delegation_memory']
            print(f"  Memory patterns: {memory_info.get('patterns_learned', 0)}")
            print(f"  Memory success rate: {memory_info.get('success_rate', 0):.1%}")
        
        # Save report to file
        report_file = Path.home() / '.neuralsync' / 'test_report.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests / total_tests,
                    'total_duration': time.time() - self.start_time
                },
                'test_results': self.test_results,
                'system_performance': performance
            }, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Overall assessment
        overall_success = passed_tests >= total_tests * 0.8  # 80% pass rate
        
        print(f"\n{'🎉' if overall_success else '⚠️'} Overall Assessment:")
        if overall_success:
            print("  The NeuralSync v2 orchestration system is working well!")
            print("  Claude Code can successfully delegate to codex and gemini as extensions.")
        else:
            print("  The system needs some adjustments for optimal performance.")
            print("  Check the detailed results and fix failing components.")
        
        return overall_success


async def main():
    """Main test function"""
    print("Starting NeuralSync v2 Orchestration System Tests...")
    
    tester = OrchestrationTester()
    success = await tester.run_all_tests()
    
    # Show delegation memory analysis
    print(f"\n🧠 Delegation Memory Analysis:")
    await analyze_delegation_effectiveness()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))