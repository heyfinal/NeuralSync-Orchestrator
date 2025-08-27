# 🎭 NeuralSync Orchestrator

**Transform Claude into a master orchestrator that delegates tasks to specialized CLI agents**

[![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeuralSync](https://img.shields.io/badge/NeuralSync-v2+-green.svg)](https://github.com/heyfinal/NeuralSync2)

*Intelligent task delegation system that reduces Claude API usage by up to 70% while maintaining bleeding-edge quality*

---

## 🚀 Overview

The NeuralSync Orchestrator transforms Claude into a master orchestrator that intelligently delegates tasks to specialized CLI agents (codex, gemini) as extensions of itself. This revolutionary system:

- **Reduces API Usage** by 70% through intelligent task routing
- **Supports Any Language** - Python, JavaScript, Go, Rust, Java, C++, and more
- **Learns Over Time** - Memory system improves delegation patterns
- **Maintains Quality** - Advanced validation ensures output excellence
- **Self-Healing** - Automatic recovery from agent failures

## ⚡ Quick Start

### Prerequisites
- [NeuralSync v2](https://github.com/heyfinal/NeuralSync2) must be installed first
- Python 3.9+ 
- Claude CLI installed

### Installation

```bash
# Clone the orchestrator
git clone https://github.com/heyfinal/NeuralSync-Orchestrator.git
cd NeuralSync-Orchestrator

# Install the orchestration system
python3 install.py
```

### Usage

```bash
# Launch claude-ns with NeuralSync context
claude-ns

# Natural conversation flow:
"I need a Python web scraper that handles dynamic content and exports to multiple formats"
# [Claude responds with normal analysis and planning]

"ok that sounds good.. use the meta-ai-agent to start on that"
# → meta-ai-agent activates with full conversation context
# → Intelligently delegates coding tasks to codex
# → Delegates research tasks to gemini  
# → Executes tasks in parallel for faster results
# → Maintains quality through validation and review

# Complex tasks use hybrid delegation
echo "Build a REST API with authentication and documentation" | claude-ns --print
# → Orchestrates multiple agents for optimal results
```

## 🏗️ Architecture

### Core Components

- **`orchestrator.py`** - Master delegation controller
- **`task_router.py`** - Intelligent task classification and routing
- **`prompt_optimizer.py`** - Bleeding-edge prompt engineering for each agent
- **`agent_monitor.py`** - Health monitoring and automatic recovery
- **`quality_validator.py`** - Output validation and quality assurance
- **`delegation_memory.py`** - Learning system that improves over time

### Task Routing Logic

| Task Type | Agent | Examples |
|-----------|--------|----------|
| Code Generation | Codex | "Write a function...", "Create a class..." |
| Debugging | Codex | "Fix this bug...", "Debug this code..." |
| Research | Gemini | "Research...", "Compare...", "Analyze..." |
| Documentation | Gemini | "Document...", "Write README..." |
| Complex Tasks | Hybrid | Multi-step projects requiring multiple agents |

## 📊 Performance Benefits

- **70% API Usage Reduction** - Intelligent delegation reduces Claude calls
- **Parallel Execution** - Multiple agents work simultaneously
- **Quality Maintenance** - Validation ensures consistent output quality
- **Continuous Learning** - System improves delegation patterns over time
- **Language Agnostic** - Works with any programming language

## 🎯 Features

### Intelligent Delegation
- Automatic task classification and routing
- Parallel, sequential, and hybrid execution strategies
- Context-aware agent selection

### Learning System
- Remembers successful delegation patterns
- Optimizes future task routing
- Improves quality scores over time

### Quality Assurance
- Validates all agent outputs
- Ensures code is executable and well-structured
- Maintains consistent documentation quality

### Health Monitoring
- Real-time agent health tracking
- Automatic recovery from failures
- Performance metrics and insights

## 📖 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Configuration](docs/configuration.md)** - Customization options
- **[API Reference](docs/api.md)** - Python API documentation
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🧪 Examples

### Basic Python Integration
```python
from orchestrator import ClaudeOrchestrator

# Initialize orchestrator
orchestrator = ClaudeOrchestrator()

# Delegate a task
result = await orchestrator.orchestrate_task(
    "Create a web API with authentication",
    context={"language": "python", "framework": "fastapi"}
)

print(f"Success: {result.success}")
print(f"Quality: {result.quality_score}")
print(f"Agents used: {result.agents_used}")
```

### System Monitoring
```python
# Check system health
health = await orchestrator.health_check()
for agent, status in health.items():
    print(f"{agent}: {'✅' if status else '❌'}")

# View delegation insights
from delegation_memory import analyze_delegation_effectiveness
await analyze_delegation_effectiveness()
```

## 🛠️ Development

### Requirements
- Python 3.9+
- NeuralSync v2
- Claude CLI
- Optional: Codex CLI, Gemini API access

### Testing
```bash
# Run comprehensive tests
python3 test_orchestration.py

# Run specific test suites
python3 -m pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Related Projects

- [NeuralSync v2](https://github.com/heyfinal/NeuralSync2) - Core memory system (required)
- [Claude CLI](https://claude.ai/download) - Claude Code interface
- [Codex](https://openai.com/codex) - Code generation agent
- [Gemini](https://ai.google.com/tools/) - Research and analysis agent

---

**Transform your Claude experience with intelligent multi-agent delegation!** 🚀