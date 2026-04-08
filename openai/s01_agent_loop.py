import os
import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.environ["OPENAI_MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the current workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    }
}]


@dataclass
class LoopState:
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None

def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


def execute_tool_calls(tool_calls):
    results = []

    for call in tool_calls:
        name = call.function.name

        if name == "bash":
            args = json.loads(call.function.arguments)

            cmd = args.get("command")
            print(f"\033[33m$ {cmd}\033[0m")

            output = run_bash(cmd)
            print(output[:200])

            results.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })

    return results

def run_one_turn(state: LoopState):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            *state.messages
        ],
        tools=TOOLS,
        tool_choice="auto"
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    msg = choice.message

    assistant_msg = {"role": "assistant", "content": msg.content}

    if msg.tool_calls:
        assistant_msg["tool_calls"] = msg.tool_calls
    state.messages.append(assistant_msg)

    if finish_reason != "tool_calls":
        if msg.content:
            print(f"\033[32m{msg.content}\033[0m")
        state.transition_reason = None
        return False

    results = execute_tool_calls(msg.tool_calls)
    if not results:
        state.transition_reason = None
        return False

    for r in results:
        state.messages.append(r)

    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True


def agent_loop(state: LoopState):
    while run_one_turn(state):
        pass

if __name__ == "__main__":
    history = []

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})

        state = LoopState(messages=history)
        agent_loop(state)
        
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()