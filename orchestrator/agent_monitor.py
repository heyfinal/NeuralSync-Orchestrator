#!/usr/bin/env python3
"""
Agent Health Monitor and Recovery System
Monitors CLI agent health and provides recovery mechanisms
"""

import asyncio
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AgentHealth:
    agent: str
    status: str  # healthy, degraded, unhealthy, offline
    response_time: float
    success_rate: float
    last_check: float
    error_count: int
    recovery_attempts: int

class AgentHealthMonitor:
    """Monitors and manages health of CLI agents"""
    
    def __init__(self):
        self.health_status: Dict[str, AgentHealth] = {}
        self.health_check_interval = 30.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Health thresholds
        self.response_time_threshold = 30.0  # seconds
        self.success_rate_threshold = 0.8
        self.max_recovery_attempts = 3
        
        # Initialize health status
        for agent in ['codex', 'gemini', 'claude']:
            self.health_status[agent] = AgentHealth(
                agent=agent,
                status='unknown',
                response_time=0.0,
                success_rate=1.0,
                last_check=0.0,
                error_count=0,
                recovery_attempts=0
            )
    
    def is_healthy(self) -> bool:
        """Health check for the monitor itself"""
        return True
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Agent health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_all_agents()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_agents(self):
        """Check health of all agents"""
        for agent in self.health_status.keys():
            await self._check_agent_health(agent)
    
    async def is_agent_healthy(self, agent: str) -> bool:
        """Check if specific agent is healthy"""
        if agent not in self.health_status:
            return False
        
        health = self.health_status[agent]
        
        # Force refresh if data is stale
        if time.time() - health.last_check > 60:
            await self._check_agent_health(agent)
            health = self.health_status[agent]
        
        return health.status in ['healthy', 'degraded']
    
    async def _check_agent_health(self, agent: str):
        """Perform health check on specific agent"""
        health = self.health_status[agent]
        start_time = time.time()
        
        try:
            if agent == 'codex':
                success = await self._check_codex_health()
            elif agent == 'gemini':
                success = await self._check_gemini_health()
            elif agent == 'claude':
                success = await self._check_claude_health()
            else:
                success = False
            
            response_time = time.time() - start_time
            
            # Update health status
            health.last_check = time.time()
            health.response_time = response_time
            
            if success:
                health.error_count = max(0, health.error_count - 1)  # Recover from errors
                if response_time <= self.response_time_threshold:
                    health.status = 'healthy'
                else:
                    health.status = 'degraded'
            else:
                health.error_count += 1
                if health.error_count >= 5:
                    health.status = 'unhealthy'
                elif health.error_count >= 2:
                    health.status = 'degraded'
            
            # Update success rate (rolling average)
            current_success = 1.0 if success else 0.0
            health.success_rate = (health.success_rate * 0.8) + (current_success * 0.2)
            
        except Exception as e:
            logger.error(f"Health check failed for {agent}: {e}")
            health.status = 'offline'
            health.error_count += 1
            health.last_check = time.time()
    
    async def _check_codex_health(self) -> bool:
        """Check codex CLI health"""
        try:
            process = await asyncio.create_subprocess_exec(
                'codex', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
            
        except (asyncio.TimeoutError, FileNotFoundError):
            return False
        except Exception as e:
            logger.debug(f"Codex health check error: {e}")
            return False
    
    async def _check_gemini_health(self) -> bool:
        """Check gemini CLI health"""
        try:
            process = await asyncio.create_subprocess_exec(
                'gemini', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
            
        except (asyncio.TimeoutError, FileNotFoundError):
            return False
        except Exception as e:
            logger.debug(f"Gemini health check error: {e}")
            return False
    
    async def _check_claude_health(self) -> bool:
        """Check Claude health (always healthy since we're running in it)"""
        return True
    
    async def recover_agent(self, agent: str) -> bool:
        """Attempt to recover an unhealthy agent"""
        health = self.health_status[agent]
        
        if health.recovery_attempts >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for {agent}")
            return False
        
        health.recovery_attempts += 1
        logger.info(f"Attempting recovery for {agent} (attempt {health.recovery_attempts})")
        
        try:
            if agent == 'codex':
                return await self._recover_codex()
            elif agent == 'gemini':
                return await self._recover_gemini()
            elif agent == 'claude':
                return True  # Claude doesn't need recovery
            
        except Exception as e:
            logger.error(f"Recovery failed for {agent}: {e}")
        
        return False
    
    async def _recover_codex(self) -> bool:
        """Attempt to recover codex CLI"""
        try:
            # Try a simple command to wake up codex
            process = await asyncio.create_subprocess_exec(
                'codex', '--help',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(process.communicate(), timeout=15)
            
            # Wait a moment then check health
            await asyncio.sleep(2)
            return await self._check_codex_health()
            
        except Exception as e:
            logger.error(f"Codex recovery error: {e}")
            return False
    
    async def _recover_gemini(self) -> bool:
        """Attempt to recover gemini CLI"""
        try:
            # Check if API key is set
            import os
            if not os.environ.get('GOOGLE_API_KEY'):
                logger.warning("GOOGLE_API_KEY not set - gemini may not work")
                return False
            
            # Try a simple command
            process = await asyncio.create_subprocess_exec(
                'python3', '-c', 'import google.generativeai; print("OK")',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode == 0:
                await asyncio.sleep(1)
                return await self._check_gemini_health()
            
            return False
            
        except Exception as e:
            logger.error(f"Gemini recovery error: {e}")
            return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        summary = {
            'overall_status': 'healthy',
            'agents': {},
            'monitoring': self.is_monitoring,
            'last_update': max(h.last_check for h in self.health_status.values()) if self.health_status else 0
        }
        
        unhealthy_count = 0
        
        for agent, health in self.health_status.items():
            agent_summary = {
                'status': health.status,
                'response_time': health.response_time,
                'success_rate': health.success_rate,
                'error_count': health.error_count,
                'recovery_attempts': health.recovery_attempts,
                'last_check_ago': time.time() - health.last_check if health.last_check else 0
            }
            
            summary['agents'][agent] = agent_summary
            
            if health.status in ['unhealthy', 'offline']:
                unhealthy_count += 1
        
        # Determine overall status
        total_agents = len(self.health_status)
        if unhealthy_count == 0:
            summary['overall_status'] = 'healthy'
        elif unhealthy_count < total_agents / 2:
            summary['overall_status'] = 'degraded'
        else:
            summary['overall_status'] = 'unhealthy'
        
        return summary
    
    async def force_health_check(self, agent: str = None) -> Dict[str, Any]:
        """Force immediate health check"""
        if agent:
            if agent in self.health_status:
                await self._check_agent_health(agent)
                return {agent: self.health_status[agent]}
            else:
                return {'error': f'Unknown agent: {agent}'}
        else:
            await self._check_all_agents()
            return self.get_health_summary()