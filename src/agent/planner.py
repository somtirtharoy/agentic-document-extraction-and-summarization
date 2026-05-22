from collections.abc import Callable
from pathlib import Path

import yaml
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    Part,
)

from src.agent.memory import AgentMemory
from src.agent.schemas import TOOL_DECLARATIONS
from src.agent.tools import AgentTools
from src.gcp.vertex_client import get_model
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "config" / "prompts" / "agent_system.yaml"
MAX_STEPS = 8


def _load_system_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["system"]


def _history_to_contents(history: list[dict]) -> list[Content]:
    """Reconstruct Vertex AI Content objects from Firestore-persisted history."""
    contents: list[Content] = []
    for turn in history:
        role = turn.get("role")
        if role == "user":
            contents.append(Content(role="user", parts=[Part.from_text(turn["content"])]))
        elif role == "model":
            contents.append(Content(role="model", parts=[Part.from_text(turn["content"])]))
        elif role == "tool":
            # Re-emit as model function_call + user function_response pair
            contents.append(
                Content(
                    role="model",
                    parts=[Part.from_function_call(
                        name=turn["tool_name"],
                        args=turn.get("args", {}),
                    )],
                )
            )
            contents.append(
                Content(
                    role="user",
                    parts=[Part.from_function_response(
                        name=turn["tool_name"],
                        response={"result": turn.get("observation", {})},
                    )],
                )
            )
    return contents


class Planner:
    def __init__(
        self,
        tools: AgentTools,
        memory: AgentMemory,
        trace_callback: Callable[[int, str, dict, dict], None] | None = None,
    ) -> None:
        self._tools = tools
        self._memory = memory
        self._model = get_model()
        self._system_prompt = _load_system_prompt()
        self._generation_config = GenerationConfig(temperature=0.2, max_output_tokens=2048)
        self._trace_callback = trace_callback  # called each step for REPL trace output

    def run(self, session_id: str, user_message: str) -> str:
        """Run the ReAct loop for one user turn. Returns the final answer."""
        # Persist user message
        self._memory.append_user(session_id, user_message)

        # Reconstruct conversation context
        history = self._memory.get_history(session_id)
        contents = _history_to_contents(history)

        for step in range(1, MAX_STEPS + 1):
            response = self._model.generate_content(
                contents,
                generation_config=self._generation_config,
                tools=[TOOL_DECLARATIONS],
                system_instruction=self._system_prompt,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts

            # Check if the model wants to call a tool
            function_calls = [p for p in parts if p.function_call and p.function_call.name]

            if function_calls:
                fc = function_calls[0].function_call
                tool_name = fc.name
                tool_args = dict(fc.args)

                logger.info(
                    "Agent tool call",
                    extra={"step": step, "tool": tool_name, "args": tool_args},
                )

                # Execute the tool
                observation = self._tools.dispatch(tool_name, tool_args)

                # Fire trace callback for REPL display
                if self._trace_callback:
                    self._trace_callback(step, tool_name, tool_args, observation)

                # Persist tool turn
                self._memory.append_tool(session_id, tool_name, tool_args, observation)

                # Append model's function call + tool response to contents
                contents.append(candidate.content)
                contents.append(
                    Content(
                        role="user",
                        parts=[Part.from_function_response(
                            name=tool_name,
                            response={"result": observation},
                        )],
                    )
                )

            else:
                # No function call → final text answer
                final_answer = "".join(p.text for p in parts if hasattr(p, "text"))
                self._memory.append_model(session_id, final_answer)
                logger.info("Agent returned final answer", extra={"steps": step})
                return final_answer

        timeout_msg = (
            "I was unable to complete the task within the allowed number of steps. "
            "Please try rephrasing your question or narrowing the scope."
        )
        self._memory.append_model(session_id, timeout_msg)
        return timeout_msg
