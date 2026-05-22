import uuid
from collections.abc import Callable

from src.agent.memory import AgentMemory
from src.agent.planner import Planner
from src.agent.tools import AgentTools
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Agent:
    """Top-level Research Insight Agent.

    Usage:
        agent = Agent()
        session_id = agent.new_session()
        answer = agent.chat(session_id, "What are the top entities in climate articles?")
    """

    def __init__(
        self,
        trace_callback: Callable[[int, str, dict, dict], None] | None = None,
    ) -> None:
        self._memory = AgentMemory()
        self._tools = AgentTools(memory=self._memory)
        self._planner = Planner(
            tools=self._tools,
            memory=self._memory,
            trace_callback=trace_callback,
        )

    def new_session(self) -> str:
        """Generate and return a new session ID."""
        session_id = str(uuid.uuid4())
        logger.info("New agent session", extra={"session_id": session_id})
        return session_id

    def chat(self, session_id: str, message: str) -> str:
        """Send a message and return the agent's response."""
        logger.info("User message", extra={"session_id": session_id, "user_message": message[:100]})
        return self._planner.run(session_id, message)
