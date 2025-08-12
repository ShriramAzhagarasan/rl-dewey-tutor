"""
Dynamic Task Allocation System

This module implements sophisticated task allocation between multiple RL agents,
allowing different agents to specialize in different aspects of the tutoring task
and dynamically assigning tasks based on agent expertise and current context.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque

class TaskType(Enum):
    """Different types of tutoring tasks"""
    EXPLORATION = "exploration"          # Finding optimal difficulty for new topics
    EXPLOITATION = "exploitation"        # Maintaining performance on known topics
    ADAPTATION = "adaptation"           # Adjusting to changing student performance
    RECOVERY = "recovery"               # Helping struggling students
    ACCELERATION = "acceleration"       # Advancing high-performing students
    ASSESSMENT = "assessment"           # Evaluating current skill levels
    REMEDIATION = "remediation"         # Addressing skill gaps
    CONSOLIDATION = "consolidation"     # Reinforcing learned material

class AgentSpecialization(Enum):
    """Agent specialization areas"""
    GENERAL_PURPOSE = "general"         # Good at all tasks
    EXPLORATION_SPECIALIST = "explorer" # Best at exploration tasks
    EXPLOITATION_SPECIALIST = "exploiter" # Best at exploitation tasks
    ADAPTATION_SPECIALIST = "adapter"   # Best at adaptation tasks
    STABILITY_SPECIALIST = "stabilizer" # Best at maintaining stability

@dataclass
class Task:
    """Represents a specific tutoring task"""
    task_id: str
    task_type: TaskType
    priority: float
    context: Dict[str, Any]
    requirements: Dict[str, Any]
    estimated_duration: int
    deadline: Optional[float] = None
    assigned_agent: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    success_metric: Optional[float] = None

@dataclass
class AgentCapability:
    """Represents an agent's capabilities for different tasks"""
    agent_id: str
    specialization: AgentSpecialization
    task_expertise: Dict[TaskType, float]  # 0-1 score for each task type
    current_load: float
    max_load: float
    recent_performance: Dict[TaskType, List[float]]
    availability: bool = True

class DynamicTaskAllocator:
    """
    Dynamic task allocation system for multi-agent RL tutoring
    
    Features:
    - Task decomposition and prioritization
    - Agent specialization tracking
    - Dynamic load balancing
    - Performance-based allocation
    - Context-aware task assignment
    - Real-time adaptation
    """
    
    def __init__(self, 
                 allocation_strategy: str = "expertise_weighted",
                 load_balancing: bool = True,
                 performance_weight: float = 0.6,
                 availability_weight: float = 0.3,
                 load_weight: float = 0.1):
        """
        Initialize dynamic task allocator
        
        Args:
            allocation_strategy: Strategy for task allocation
            load_balancing: Whether to balance load across agents
            performance_weight: Weight for agent performance in allocation
            availability_weight: Weight for agent availability
            load_weight: Weight for current load in allocation
        """
        self.allocation_strategy = allocation_strategy
        self.load_balancing = load_balancing
        self.performance_weight = performance_weight
        self.availability_weight = availability_weight
        self.load_weight = load_weight
        
        # Task and agent tracking
        self.pending_tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.task_queue = deque()
        
        # Performance tracking
        self.allocation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Adaptive parameters
        self.task_type_priorities = {task_type: 1.0 for task_type in TaskType}
        self.agent_efficiency_history: Dict[str, List[float]] = defaultdict(list)
        
    def register_agent(self, 
                      agent_id: str,
                      specialization: AgentSpecialization,
                      initial_expertise: Optional[Dict[TaskType, float]] = None,
                      max_load: float = 1.0):
        """
        Register a new agent with the task allocator
        
        Args:
            agent_id: Unique identifier for the agent
            specialization: Agent's specialization area
            initial_expertise: Initial expertise scores for different tasks
            max_load: Maximum load the agent can handle
        """
        
        if initial_expertise is None:
            # Default expertise based on specialization
            initial_expertise = self._get_default_expertise(specialization)
        
        capability = AgentCapability(
            agent_id=agent_id,
            specialization=specialization,
            task_expertise=initial_expertise,
            current_load=0.0,
            max_load=max_load,
            recent_performance=defaultdict(list)
        )
        
        self.agent_capabilities[agent_id] = capability
        print(f"✅ Registered agent {agent_id} with specialization {specialization.value}")
    
    def submit_task(self, 
                   task_type: TaskType,
                   context: Dict[str, Any],
                   priority: float = 1.0,
                   requirements: Optional[Dict[str, Any]] = None,
                   estimated_duration: int = 1,
                   deadline: Optional[float] = None) -> str:
        """
        Submit a new task for allocation
        
        Args:
            task_type: Type of the task
            context: Context information for the task
            priority: Task priority (higher = more important)
            requirements: Specific requirements for task execution
            estimated_duration: Estimated duration in time steps
            deadline: Optional deadline for task completion
            
        Returns:
            Task ID
        """
        
        task_id = f"task_{len(self.pending_tasks) + len(self.active_tasks) + len(self.completed_tasks)}_{time.time()}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            context=context,
            requirements=requirements or {},
            estimated_duration=estimated_duration,
            deadline=deadline
        )
        
        self.pending_tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Try immediate allocation
        self._allocate_pending_tasks()
        
        return task_id
    
    def allocate_tasks_for_step(self, 
                               student_state: Dict[str, Any],
                               environment_context: Dict[str, Any]) -> Dict[str, str]:
        """
        Allocate tasks for the current step based on student state and context
        
        Args:
            student_state: Current student state information
            environment_context: Current environment context
            
        Returns:
            Dictionary mapping agent_id to task_id for this step
        """
        
        # Analyze current situation and generate tasks
        new_tasks = self._analyze_and_generate_tasks(student_state, environment_context)
        
        # Submit generated tasks
        for task_info in new_tasks:
            self.submit_task(**task_info)
        
        # Allocate all pending tasks
        allocations = self._allocate_pending_tasks()
        
        # Update agent loads
        self._update_agent_loads()
        
        return allocations
    
    def complete_task(self, 
                     task_id: str, 
                     success_metric: float,
                     agent_performance: Dict[str, Any]):
        """
        Mark a task as completed and update agent performance
        
        Args:
            task_id: ID of the completed task
            success_metric: Success metric for the task (0-1)
            agent_performance: Additional performance metrics
        """
        
        if task_id not in self.active_tasks:
            print(f"⚠️  Task {task_id} not found in active tasks")
            return
        
        task = self.active_tasks[task_id]
        task.completion_time = time.time()
        task.success_metric = success_metric
        
        # Update agent performance
        agent_id = task.assigned_agent
        if agent_id and agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            capability.recent_performance[task.task_type].append(success_metric)
            
            # Keep only recent performance (last 10 tasks per type)
            if len(capability.recent_performance[task.task_type]) > 10:
                capability.recent_performance[task.task_type] = \
                    capability.recent_performance[task.task_type][-10:]
            
            # Update expertise based on performance
            self._update_agent_expertise(agent_id, task.task_type, success_metric)
            
            # Reduce agent load
            capability.current_load = max(0, capability.current_load - task.estimated_duration)
        
        # Move task to completed
        self.completed_tasks[task_id] = task
        del self.active_tasks[task_id]
        
        print(f"✅ Task {task_id} completed by {agent_id} with success {success_metric:.3f}")
    
    def fail_task(self, task_id: str, reason: str):
        """Mark a task as failed"""
        
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task.completion_time = time.time()
        task.success_metric = 0.0
        
        # Update agent performance (failure)
        agent_id = task.assigned_agent
        if agent_id and agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            capability.recent_performance[task.task_type].append(0.0)
            
            # Reduce agent load
            capability.current_load = max(0, capability.current_load - task.estimated_duration)
        
        # Move task to failed
        self.failed_tasks[task_id] = task
        del self.active_tasks[task_id]
        
        print(f"❌ Task {task_id} failed: {reason}")
    
    def get_agent_assignment(self, agent_id: str) -> Optional[str]:
        """Get current task assignment for an agent"""
        
        for task_id, task in self.active_tasks.items():
            if task.assigned_agent == agent_id:
                return task_id
        
        return None
    
    def get_task_allocation_stats(self) -> Dict[str, Any]:
        """Get comprehensive task allocation statistics"""
        
        stats = {
            'task_counts': {
                'pending': len(self.pending_tasks),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks)
            },
            'agent_utilization': {},
            'task_type_distribution': defaultdict(int),
            'success_rates': {},
            'average_completion_time': {},
            'load_balance_metric': self._calculate_load_balance()
        }
        
        # Agent utilization
        for agent_id, capability in self.agent_capabilities.items():
            stats['agent_utilization'][agent_id] = {
                'current_load': capability.current_load,
                'max_load': capability.max_load,
                'utilization_rate': capability.current_load / capability.max_load,
                'availability': capability.availability
            }
        
        # Task type distribution
        for task in list(self.active_tasks.values()) + list(self.completed_tasks.values()):
            stats['task_type_distribution'][task.task_type.value] += 1
        
        # Success rates by agent
        for agent_id, capability in self.agent_capabilities.items():
            total_tasks = 0
            total_success = 0
            
            for task_type, performance_list in capability.recent_performance.items():
                total_tasks += len(performance_list)
                total_success += sum(performance_list)
            
            if total_tasks > 0:
                stats['success_rates'][agent_id] = total_success / total_tasks
            else:
                stats['success_rates'][agent_id] = 0.0
        
        # Average completion times by task type
        completion_times = defaultdict(list)
        for task in self.completed_tasks.values():
            if task.start_time and task.completion_time:
                duration = task.completion_time - task.start_time
                completion_times[task.task_type.value].append(duration)
        
        for task_type, times in completion_times.items():
            stats['average_completion_time'][task_type] = np.mean(times)
        
        return stats
    
    def _analyze_and_generate_tasks(self, 
                                  student_state: Dict[str, Any],
                                  environment_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze current situation and generate appropriate tasks"""
        
        tasks = []
        
        # Extract key information
        skill_levels = student_state.get('skill_levels', [])
        topic_mastery = student_state.get('topic_mastery', [])
        confidence = student_state.get('confidence', 0.5)
        last_performance = student_state.get('last_performance', 0.5)
        
        # Current difficulty levels being used
        current_difficulties = environment_context.get('current_difficulties', [])
        
        # Generate tasks based on analysis
        
        # 1. Exploration tasks for low-confidence areas
        if confidence < 0.3:
            tasks.append({
                'task_type': TaskType.EXPLORATION,
                'context': {'low_confidence': True, 'confidence': confidence},
                'priority': 0.8,
                'estimated_duration': 2
            })
        
        # 2. Adaptation tasks for performance mismatches
        if skill_levels and current_difficulties:
            for i, (skill, difficulty) in enumerate(zip(skill_levels, current_difficulties)):
                difficulty_norm = difficulty / (len(current_difficulties) - 1) if len(current_difficulties) > 1 else 0
                if abs(skill - difficulty_norm) > 0.3:  # Significant mismatch
                    tasks.append({
                        'task_type': TaskType.ADAPTATION,
                        'context': {'topic': i, 'skill_level': skill, 'current_difficulty': difficulty_norm},
                        'priority': 0.9,
                        'estimated_duration': 1
                    })
        
        # 3. Recovery tasks for poor performance
        if last_performance < 0.3:
            tasks.append({
                'task_type': TaskType.RECOVERY,
                'context': {'poor_performance': True, 'performance': last_performance},
                'priority': 1.0,
                'estimated_duration': 3
            })
        
        # 4. Acceleration tasks for high performance
        if last_performance > 0.8 and confidence > 0.7:
            tasks.append({
                'task_type': TaskType.ACCELERATION,
                'context': {'high_performance': True, 'performance': last_performance},
                'priority': 0.6,
                'estimated_duration': 1
            })
        
        # 5. Assessment tasks periodically
        if len(self.completed_tasks) % 20 == 0:  # Every 20 tasks
            tasks.append({
                'task_type': TaskType.ASSESSMENT,
                'context': {'periodic_assessment': True},
                'priority': 0.5,
                'estimated_duration': 1
            })
        
        # 6. Consolidation tasks for mastered topics
        if topic_mastery:
            for i, mastery in enumerate(topic_mastery):
                if mastery > 0.8:
                    tasks.append({
                        'task_type': TaskType.CONSOLIDATION,
                        'context': {'topic': i, 'mastery_level': mastery},
                        'priority': 0.4,
                        'estimated_duration': 1
                    })
        
        return tasks
    
    def _allocate_pending_tasks(self) -> Dict[str, str]:
        """Allocate all pending tasks to available agents"""
        
        allocations = {}
        
        # Sort tasks by priority
        sorted_task_ids = sorted(
            self.pending_tasks.keys(),
            key=lambda tid: self.pending_tasks[tid].priority,
            reverse=True
        )
        
        for task_id in sorted_task_ids:
            task = self.pending_tasks[task_id]
            
            # Find best agent for this task
            best_agent = self._find_best_agent_for_task(task)
            
            if best_agent:
                # Allocate task to agent
                task.assigned_agent = best_agent
                task.start_time = time.time()
                
                # Update agent load
                capability = self.agent_capabilities[best_agent]
                capability.current_load += task.estimated_duration
                
                # Move task to active
                self.active_tasks[task_id] = task
                del self.pending_tasks[task_id]
                
                allocations[best_agent] = task_id
                
                # Record allocation
                self.allocation_history.append({
                    'timestamp': time.time(),
                    'task_id': task_id,
                    'task_type': task.task_type.value,
                    'agent_id': best_agent,
                    'priority': task.priority
                })
        
        return allocations
    
    def _find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best available agent for a specific task"""
        
        available_agents = [
            agent_id for agent_id, capability in self.agent_capabilities.items()
            if capability.availability and capability.current_load + task.estimated_duration <= capability.max_load
        ]
        
        if not available_agents:
            return None
        
        # Calculate scores for each agent
        agent_scores = {}
        
        for agent_id in available_agents:
            capability = self.agent_capabilities[agent_id]
            
            # Performance score (expertise in this task type)
            expertise_score = capability.task_expertise.get(task.task_type, 0.5)
            
            # Recent performance score
            recent_performances = capability.recent_performance.get(task.task_type, [])
            if recent_performances:
                performance_score = np.mean(recent_performances)
            else:
                performance_score = 0.5  # Default
            
            # Availability score (inverse of current load)
            load_ratio = capability.current_load / capability.max_load
            availability_score = 1.0 - load_ratio
            
            # Combined score
            total_score = (
                self.performance_weight * (expertise_score + performance_score) / 2 +
                self.availability_weight * availability_score +
                self.load_weight * (1.0 - load_ratio)
            )
            
            agent_scores[agent_id] = total_score
        
        # Return agent with highest score
        best_agent = max(agent_scores.keys(), key=lambda aid: agent_scores[aid])
        return best_agent
    
    def _get_default_expertise(self, specialization: AgentSpecialization) -> Dict[TaskType, float]:
        """Get default expertise scores based on agent specialization"""
        
        # Base expertise (all agents have some capability in all tasks)
        base_expertise = {task_type: 0.5 for task_type in TaskType}
        
        # Specialization bonuses
        if specialization == AgentSpecialization.EXPLORATION_SPECIALIST:
            base_expertise[TaskType.EXPLORATION] = 0.9
            base_expertise[TaskType.ASSESSMENT] = 0.8
            base_expertise[TaskType.ADAPTATION] = 0.7
        
        elif specialization == AgentSpecialization.EXPLOITATION_SPECIALIST:
            base_expertise[TaskType.EXPLOITATION] = 0.9
            base_expertise[TaskType.CONSOLIDATION] = 0.8
            base_expertise[TaskType.ACCELERATION] = 0.7
        
        elif specialization == AgentSpecialization.ADAPTATION_SPECIALIST:
            base_expertise[TaskType.ADAPTATION] = 0.9
            base_expertise[TaskType.RECOVERY] = 0.8
            base_expertise[TaskType.REMEDIATION] = 0.7
        
        elif specialization == AgentSpecialization.STABILITY_SPECIALIST:
            base_expertise[TaskType.CONSOLIDATION] = 0.9
            base_expertise[TaskType.EXPLOITATION] = 0.8
            base_expertise[TaskType.ASSESSMENT] = 0.7
        
        else:  # GENERAL_PURPOSE
            # Balanced expertise across all tasks
            for task_type in TaskType:
                base_expertise[task_type] = 0.7
        
        return base_expertise
    
    def _update_agent_expertise(self, agent_id: str, task_type: TaskType, success_metric: float):
        """Update agent expertise based on task performance"""
        
        capability = self.agent_capabilities[agent_id]
        current_expertise = capability.task_expertise[task_type]
        
        # Adaptive learning rate based on current expertise
        learning_rate = 0.1 * (1.0 - current_expertise) if success_metric > 0.5 else 0.05
        
        # Update expertise with exponential moving average
        new_expertise = current_expertise + learning_rate * (success_metric - current_expertise)
        capability.task_expertise[task_type] = np.clip(new_expertise, 0.0, 1.0)
    
    def _update_agent_loads(self):
        """Update agent loads and availability"""
        
        current_time = time.time()
        
        for agent_id, capability in self.agent_capabilities.items():
            # Decay load over time (agents complete tasks)
            time_decay = 0.1  # 10% load reduction per time step
            capability.current_load = max(0, capability.current_load - time_decay)
            
            # Update availability based on load
            capability.availability = capability.current_load < capability.max_load
    
    def _calculate_load_balance(self) -> float:
        """Calculate load balance metric across all agents"""
        
        if not self.agent_capabilities:
            return 1.0
        
        loads = [cap.current_load / cap.max_load for cap in self.agent_capabilities.values()]
        
        if len(loads) <= 1:
            return 1.0
        
        # Load balance as inverse of standard deviation
        load_std = np.std(loads)
        balance_metric = 1.0 / (1.0 + load_std)
        
        return float(balance_metric)
    
    def optimize_allocation_strategy(self):
        """Optimize allocation strategy based on historical performance"""
        
        if len(self.allocation_history) < 50:
            return  # Need sufficient history
        
        # Analyze allocation performance
        recent_allocations = self.allocation_history[-50:]
        
        # Calculate success rates by task type and agent specialization
        success_by_specialization = defaultdict(list)
        
        for allocation in recent_allocations:
            task_id = allocation['task_id']
            agent_id = allocation['agent_id']
            
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.success_metric is not None:
                    agent_spec = self.agent_capabilities[agent_id].specialization
                    success_by_specialization[agent_spec].append(task.success_metric)
        
        # Update task type priorities based on success rates
        for specialization, success_rates in success_by_specialization.items():
            if success_rates:
                avg_success = np.mean(success_rates)
                
                # Adjust priorities for task types this specialization handles well
                if specialization == AgentSpecialization.EXPLORATION_SPECIALIST and avg_success > 0.7:
                    self.task_type_priorities[TaskType.EXPLORATION] *= 1.1
                elif specialization == AgentSpecialization.ADAPTATION_SPECIALIST and avg_success > 0.7:
                    self.task_type_priorities[TaskType.ADAPTATION] *= 1.1
                # Add more adjustments as needed
        
        # Normalize priorities
        max_priority = max(self.task_type_priorities.values())
        for task_type in self.task_type_priorities:
            self.task_type_priorities[task_type] /= max_priority
        
        print("🔄 Allocation strategy optimized based on performance history")
    
    def get_allocation_recommendations(self) -> Dict[str, str]:
        """Get recommendations for improving task allocation"""
        
        recommendations = {}
        
        # Analyze agent utilization
        utilizations = [cap.current_load / cap.max_load for cap in self.agent_capabilities.values()]
        
        if utilizations:
            avg_utilization = np.mean(utilizations)
            utilization_variance = np.var(utilizations)
            
            if avg_utilization < 0.3:
                recommendations['utilization'] = "Consider increasing task complexity or reducing number of agents"
            elif avg_utilization > 0.9:
                recommendations['utilization'] = "Consider adding more agents or simplifying tasks"
            
            if utilization_variance > 0.1:
                recommendations['load_balance'] = "Improve load balancing across agents"
        
        # Analyze success rates
        overall_success_rates = []
        for capability in self.agent_capabilities.values():
            for performance_list in capability.recent_performance.values():
                overall_success_rates.extend(performance_list)
        
        if overall_success_rates:
            avg_success = np.mean(overall_success_rates)
            if avg_success < 0.5:
                recommendations['performance'] = "Consider retraining agents or adjusting task difficulty"
        
        return recommendations
