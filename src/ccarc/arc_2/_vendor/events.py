from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class EventType(Enum):
    """Types of events emitted by the orchestrator."""

    SYSTEM = "system"
    THINKING = "thinking"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TURN_START = "turn_start"
    IMAGE = "image"
    ERROR = "error"
    REFLECTION = "reflection"
    REFLECTOR_CONTEXT = "reflector_context"
    COMPLETE = "complete"


@dataclass
class OrchestratorEvent:
    """Event emitted by the orchestrator."""

    type: EventType
    content: str = ""
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)


EventCallback = Callable[[OrchestratorEvent], None]
