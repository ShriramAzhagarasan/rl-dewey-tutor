"""
Orchestration module for RL-Dewey-Tutor

This module provides orchestration components for coordinating
multiple RL agents and managing complex multi-agent interactions.
"""

from .task_allocator import DynamicTaskAllocator, TaskType, AgentSpecialization, Task, AgentCapability

__all__ = [
    'DynamicTaskAllocator',
    'TaskType', 
    'AgentSpecialization',
    'Task',
    'AgentCapability'
]
