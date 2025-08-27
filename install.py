#!/usr/bin/env python3
"""
NeuralSync Orchestrator Installer
Transforms Claude into a master orchestrator that delegates to specialized agents
"""

import sys
import os
import subprocess
import shutil
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class OrchestratorInstaller:
    """Installer for NeuralSync Orchestrator"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.orchestrator_dir = self.base_dir / 'orchestrator'
        
        # Check for NeuralSync v2 installation
        self.neuralsync_paths = [
            Path.home() / 'NeuralSync2',
            Path.cwd() / 'NeuralSync2',
            Path('/opt/neuralsync'),
            Path.cwd().parent / 'NeuralSync2'
        ]
        self.neuralsync_dir = None
        self._find_neuralsync()
        
        self.cli_agents = {
            'codex': {
                'check_cmd': ['codex', '--version'],
                'install_cmd': ['brew', 'install', 'codex'] if sys.platform == 'darwin' else None,
                'description': 'Codex CLI for code generation'
            },
            'gemini': {
                'check_cmd': ['python3', '-c', 'import google.generativeai'],
                'install_cmd': ['pip3', 'install', 'google-generativeai'],
                'description': 'Google Generative AI for research'
            }
        }
    
    def _find_neuralsync(self):
        """Find NeuralSync v2 installation"""
        for path in self.neuralsync_paths:
            if path.exists() and (path / 'neuralsync').exists():
                self.neuralsync_dir = path
                break
    
    def print_banner(self):
        """Display installation banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║              NeuralSync Orchestrator Installer              ║
║                                                              ║
║  Transform Claude into a master orchestrator that delegates ║
║  tasks to codex and gemini as extensions of itself         ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def check_prerequisites(self) -> bool:
        """Check installation prerequisites"""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 9):
            print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.9+ required.")
            return False
        print("✅ Python version compatible")
        
        # Check NeuralSync v2
        if not self.neuralsync_dir:
            print("❌ NeuralSync v2 not found!")
            print("   Please install NeuralSync v2 first from:")
            print("   https://github.com/heyfinal/NeuralSync2")
            return False
        print(f"✅ NeuralSync v2 found at {self.neuralsync_dir}")
        
        # Check Claude CLI
        if not shutil.which('claude'):
            print("⚠️  Claude CLI not found. Please install from:")
            print("   https://claude.ai/download")
            print("   Orchestration will have limited functionality without it.")
        else:
            print("✅ Claude CLI detected")
        
        return True
    
    async def install_orchestrator_modules(self) -> bool:
        """Install orchestrator modules into NeuralSync"""
        print("\n📦 Installing orchestrator modules...")
        
        try:
            # Target directory in NeuralSync
            target_dir = self.neuralsync_dir / 'neuralsync'
            
            # Copy orchestrator modules
            modules_copied = 0
            for module_file in self.orchestrator_dir.glob('*.py'):
                if module_file.name == '__init__.py':
                    continue
                    
                target_file = target_dir / module_file.name
                
                # Handle special case for orchestrator.py -> claude_orchestrator.py
                if module_file.name == 'orchestrator.py':
                    target_file = target_dir / 'claude_orchestrator.py'
                
                # Check if file exists
                if target_file.exists():
                    response = input(f"   {target_file.name} exists. Overwrite? [y/N]: ")
                    if response.lower() != 'y':
                        print(f"   Skipping {target_file.name}")
                        continue
                
                shutil.copy2(module_file, target_file)
                modules_copied += 1
                print(f"   ✅ {target_file.name}")
            
            print(f"✅ Installed {modules_copied} orchestrator modules")
            
            # Create __init__.py if needed
            init_file = target_dir / '__init__.py'
            if not init_file.exists():
                init_file.touch()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to install modules: {e}")
            return False
    
    async def install_cli_agents(self) -> Dict[str, bool]:
        """Install CLI agents for delegation"""
        print("\n🤖 Setting up CLI agents...")
        
        results = {}
        
        for agent, config in self.cli_agents.items():
            print(f"\n   Checking {agent}...")
            
            # Check if already available
            try:
                result = subprocess.run(
                    config['check_cmd'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"   ✅ {agent} ready")
                    results[agent] = True
                    continue
            except:
                pass
            
            # Offer installation
            if config['install_cmd']:
                response = input(f"   Install {agent}? [y/N]: ")
                if response.lower() == 'y':
                    try:
                        print(f"   Installing {agent}...")
                        result = subprocess.run(
                            config['install_cmd'],
                            capture_output=True,
                            timeout=180
                        )
                        if result.returncode == 0:
                            print(f"   ✅ {agent} installed")
                            results[agent] = True
                        else:
                            print(f"   ⚠️  {agent} installation failed")
                            results[agent] = False
                    except Exception as e:
                        print(f"   ⚠️  {agent} installation error: {e}")
                        results[agent] = False
                else:
                    print(f"   Skipping {agent}")
                    results[agent] = False
            else:
                print(f"   ⚠️  No installer available for {agent}")
                results[agent] = False
        
        return results
    
    async def setup_integration(self) -> bool:
        """Set up integration with NeuralSync"""
        print("\n🔗 Setting up integration...")
        
        # Update claude-ns wrapper to support orchestration
        claude_ns_path = self.neuralsync_dir / 'bin' / 'claude-ns'
        if claude_ns_path.exists():
            print("   ✅ claude-ns wrapper found - orchestration ready")
        else:
            print("   ⚠️  claude-ns wrapper not found")
            print("       Run NeuralSync v2 installer to create wrappers")
        
        return True
    
    async def create_configuration(self) -> bool:
        """Create orchestration configuration"""
        print("\n⚙️  Creating configuration...")
        
        config_dir = Path.home() / '.neuralsync'
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / 'orchestration.json'
        
        config = {
            "enabled": True,
            "delegation": {
                "auto_delegate": False,
                "quality_threshold": 0.7,
                "fallback_to_claude": True
            },
            "agents": {
                "codex": {
                    "enabled": True,
                    "timeout": 60,
                    "retry_count": 2
                },
                "gemini": {
                    "enabled": True,
                    "timeout": 30,
                    "retry_count": 2
                }
            },
            "memory": {
                "learn_patterns": True,
                "max_patterns": 10000,
                "pattern_retention_days": 30
            }
        }
        
        # Check existing config
        if config_file.exists():
            response = input("   Configuration exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("   Keeping existing configuration")
                return True
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Configuration saved to {config_file}")
        
        # Set up API keys
        if not os.environ.get('GOOGLE_API_KEY'):
            print("\n   ⚠️  GOOGLE_API_KEY not set for Gemini")
            print("       Get your key from: https://makersuite.google.com/app/apikey")
            
        return True
    
    async def test_installation(self) -> bool:
        """Test the orchestration installation"""
        print("\n🧪 Testing installation...")
        
        try:
            # Test imports
            sys.path.insert(0, str(self.neuralsync_dir))
            
            from neuralsync.claude_orchestrator import ClaudeOrchestrator
            print("   ✅ Orchestrator imports successfully")
            
            # Test initialization
            orchestrator = ClaudeOrchestrator()
            print("   ✅ Orchestrator initializes")
            
            # Test health check
            health = await orchestrator.health_check()
            
            print("   📊 System Health:")
            for component, status in health.get('components', {}).items():
                icon = "✅" if status else "⚠️"
                print(f"       {icon} {component}")
            
            print("   🤖 Available Agents:")
            for agent, available in orchestrator.available_agents.items():
                icon = "✅" if available else "❌"
                print(f"       {icon} {agent}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    async def create_uninstaller(self) -> bool:
        """Create uninstaller"""
        print("\n📝 Creating uninstaller...")
        
        uninstaller_content = f'''#!/usr/bin/env python3
"""Uninstall NeuralSync Orchestrator"""

import os
import sys
from pathlib import Path
import shutil

def uninstall():
    print("🗑️  Uninstalling NeuralSync Orchestrator...")
    
    neuralsync_dir = Path("{self.neuralsync_dir}")
    target_dir = neuralsync_dir / "neuralsync"
    
    # Remove orchestrator modules
    modules = [
        "claude_orchestrator.py",
        "task_router.py",
        "prompt_optimizer.py", 
        "agent_monitor.py",
        "quality_validator.py",
        "delegation_memory.py"
    ]
    
    removed = 0
    for module in modules:
        module_path = target_dir / module
        if module_path.exists():
            module_path.unlink()
            removed += 1
            print(f"   ✅ Removed {{module}}")
    
    # Remove config
    config_file = Path.home() / ".neuralsync" / "orchestration.json"
    if config_file.exists():
        config_file.unlink()
        print("   ✅ Removed configuration")
    
    # Remove delegation patterns
    patterns_file = Path.home() / ".neuralsync" / "delegation_patterns.json"
    if patterns_file.exists():
        patterns_file.unlink()
        print("   ✅ Removed delegation patterns")
    
    print(f"\\n✅ Orchestrator uninstalled ({{removed}} modules removed)")
    print("   NeuralSync v2 core remains intact")

if __name__ == "__main__":
    uninstall()
'''
        
        uninstaller_path = self.base_dir / 'uninstall.py'
        with open(uninstaller_path, 'w') as f:
            f.write(uninstaller_content)
        
        uninstaller_path.chmod(0o755)
        print(f"   ✅ Uninstaller created: uninstall.py")
        
        return True
    
    async def run_installation(self) -> bool:
        """Run complete installation"""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Install modules
        if not await self.install_orchestrator_modules():
            return False
        
        # Install CLI agents
        agent_results = await self.install_cli_agents()
        
        # Setup integration
        if not await self.setup_integration():
            return False
        
        # Create configuration
        if not await self.create_configuration():
            return False
        
        # Test installation
        if not await self.test_installation():
            print("⚠️  Tests failed but installation completed")
        
        # Create uninstaller
        await self.create_uninstaller()
        
        # Success summary
        print("\n" + "="*60)
        print("🎉 NeuralSync Orchestrator Installed Successfully!")
        print("="*60)
        
        print("\n📊 Installation Summary:")
        print("   ✅ Orchestrator modules integrated with NeuralSync v2")
        print(f"   {'✅' if agent_results.get('codex') else '⚠️'} Codex: {'Ready' if agent_results.get('codex') else 'Not available'}")
        print(f"   {'✅' if agent_results.get('gemini') else '⚠️'} Gemini: {'Ready' if agent_results.get('gemini') else 'Not available'}")
        
        print("\n🚀 Quick Start:")
        print('   echo "Write a Python web scraper" | claude-ns --print')
        print('   echo "Research microservices patterns" | claude-ns --print')
        
        print("\n📖 Documentation:")
        print("   • README.md - Overview and usage")
        print("   • examples/orchestration_demo.py - Demo script")
        print("   • tests/ - Test suite")
        
        print("\n🗑️  To uninstall:")
        print("   python3 uninstall.py")
        
        return True

async def main():
    """Main installation function"""
    installer = OrchestratorInstaller()
    
    try:
        success = await installer.run_installation()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled")
        return 1
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))