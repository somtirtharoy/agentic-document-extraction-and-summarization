"""
Interactive REPL for the Research Insight Agent.
Prints the full reasoning trace (tool calls + observations) for each turn.

Usage:
    python -m scripts.05_agent_demo
    python -m scripts.05_agent_demo --session <existing-session-id>

Sample queries to try:
    - "Find articles about climate change and list the top entities"
    - "Summarise the most relevant article about the US economy"
    - "What are the recurring themes across articles about healthcare?"
    - "Compare coverage of the 2024 election across 3 articles"
"""
import argparse
import json

from src.agent.agent import Agent

# ── Trace display ─────────────────────────────────────────────────────────────

def _fmt_observation(obs: dict) -> str:
    """Pretty-print an observation dict, truncating large payloads."""
    text = json.dumps(obs, indent=2, default=str)
    lines = text.splitlines()
    if len(lines) > 20:
        return "\n".join(lines[:20]) + f"\n  ... ({len(lines) - 20} more lines)"
    return text


def _trace_callback(step: int, tool_name: str, args: dict, observation: dict) -> None:
    print(f"\n  ┌─ Step {step} ── {tool_name}")
    print(f"  │  Args        : {json.dumps(args)}")
    obs_preview = _fmt_observation(observation)
    for i, line in enumerate(obs_preview.splitlines()):
        prefix = "  │  Observation : " if i == 0 else "  │               "
        print(f"{prefix}{line}")
    print(f"  └{'─' * 50}")


# ── REPL ──────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "Find articles about climate change and list the top 5 entities mentioned.",
    "Summarise the most relevant article about the US economy.",
    "What are the recurring themes across articles about healthcare? Compare 3 articles.",
    "Find articles about elections and show me the sentiment distribution of entities.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Research Insight Agent REPL.")
    parser.add_argument("--session", type=str, default=None, help="Resume an existing session ID")
    args = parser.parse_args()

    agent = Agent(trace_callback=_trace_callback)
    session_id = args.session or agent.new_session()

    print(f"\n{'='*60}")
    print("  Research Insight Agent — Interactive Demo")
    print(f"{'='*60}")
    print(f"  Session ID : {session_id}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'samples' to see example queries.")
    print("  Type 'session' to print the current session ID.")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print(f"\nSession ended. Session ID: {session_id}")
            break

        if user_input.lower() == "samples":
            print("\nSample queries:")
            for i, q in enumerate(SAMPLE_QUERIES, 1):
                print(f"  {i}. {q}")
            print()
            continue

        if user_input.lower() == "session":
            print(f"  Session ID: {session_id}\n")
            continue

        print(f"\n{'─'*60}")
        print("  Agent reasoning:")
        print(f"{'─'*60}")

        answer = agent.chat(session_id, user_input)

        print(f"\n{'─'*60}")
        print("  Agent:")
        print(f"{'─'*60}")
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
