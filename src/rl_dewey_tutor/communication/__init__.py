"""
Communication module for RL-Dewey-Tutor

This module provides inter-agent communication and knowledge
sharing capabilities for multi-agent RL systems.
"""

from .knowledge_sharing import (
    SharedKnowledgeBase, 
    AgentCommunicationInterface,
    Experience,
    ValueEstimate,
    PolicyInsight
)

__all__ = [
    'SharedKnowledgeBase',
    'AgentCommunicationInterface',
    'Experience',
    'ValueEstimate', 
    'PolicyInsight'
]
