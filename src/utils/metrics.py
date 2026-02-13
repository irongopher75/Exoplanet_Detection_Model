from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

@dataclass
class WorkflowMetrics:
    """Collect workflow execution metrics."""
    workflow_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_step(self, step_name: str, duration: float, success: bool, extra: Optional[Dict] = None):
        """Record a workflow step."""
        step_data = {
            'name': step_name,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        }
        if extra:
            step_data.update(extra)
        self.steps.append(step_data)
        
    def add_error(self, error_msg: str):
        """Record an error message."""
        self.errors.append(error_msg)
        
    def finish(self):
        """Mark workflow as finished."""
        self.end_time = time.time()
        
    @property
    def total_duration(self) -> float:
        """Get total workflow duration."""
        end = self.end_time or time.time()
        return end - self.start_time
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'workflow_name': self.workflow_name,
            'total_duration': self.total_duration,
            'steps': self.steps,
            'errors': self.errors,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
