from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowState:
    workflow_name: str
    status: WorkflowStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_step: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.status, str):
            self.status = WorkflowStatus(self.status)

class WorkflowStateManager:
    def __init__(self, state_dir: Path = Path("outputs/workflow_state")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
    def save_state(self, state: WorkflowState) -> None:
        """Save workflow state to disk."""
        state_file = self.state_dir / f"{state.workflow_name}_state.json"
        
        # Prepare for serializing: convert status enum to string
        state_dict = asdict(state)
        state_dict['status'] = state.status.value
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
            
    def load_state(self, workflow_name: str) -> Optional[WorkflowState]:
        """Load workflow state from disk."""
        state_file = self.state_dir / f"{workflow_name}_state.json"
        if not state_file.exists():
            return None
            
        with open(state_file, 'r') as f:
            data = json.load(f)
            # Status will be converted to Enum in WorkflowState.__post_init__
            return WorkflowState(**data)
