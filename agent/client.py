"""Drives an MCP server with an open-weight model behind an OpenAI-compatible endpoint and logs every step.

python agent/client.py --server-command "python agent/server.py" --base-url URL --model NAME --task TEXT --log run.json

The API key is read from EGBENCH_LLM_API_KEY. The server's tools are offered to the model as functions, its
resources through one read_resource function, and a server prompt can open the conversation (--prompt NAME
--arg key=value). With --python-dir the model also gets a run_python function that runs code in that directory,
as a coding agent does when it trains a model of its own. The JSON log keeps the timing of every model and tool call, the token usage and the
llama-server timings of every model call, and the full reasoning, answers and tool results.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mcp import Client, MCPError, StdioServerParameters

API_KEY_VARIABLE = "EGBENCH_LLM_API_KEY"
# Recommended sampling of Qwen3 in thinking mode.
SAMPLING: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 2026,
    "max_tokens": 32768,
}
HTTP_TIMEOUT_SECONDS = 900
RESOURCE_TOOL = "read_resource"
PYTHON_TOOL = "run_python"
PYTHON_TIMEOUT_SECONDS = 1200
PYTHON_OUTPUT_CHARACTERS = 12000
# One run of the model's code must not take every core of a shared machine.
PYTHON_THREADS = "8"
SYSTEM_PROMPT = (
    "You are a research assistant. You can call the tools of an MCP server. "
    "Use the tools to obtain the information you need before you answer. "
    "Report numbers exactly as the tools return them and never invent any. "
    "Answer in the language of the user."
)
PYTHON_PROMPT = (
    " You can also run Python code with run_python, for example to train a model of your own on the data the "
    "tools return; give the tools the full paths of the files you write."
)
INSTRUCTIONS_HEADER = "The MCP server describes itself as follows:"


def _api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _open(url: str, data: bytes, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from None


def _post(url: str, body: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        return _open(url, data, headers)
    except (urllib.error.URLError, ConnectionError) as error:
        print(f"connection error ({error}), retrying once", flush=True)
        return _open(url, data, headers)


async def _list_all(fetch: Callable[..., Awaitable[Any]], field: str) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None
    while True:
        page = await fetch(cursor=cursor)
        items.extend(getattr(page, field))
        cursor = page.next_cursor
        if cursor is None:
            return items


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _resource_tool(resources: Sequence[Any], templates: Sequence[Any]) -> dict[str, Any]:
    lines = ["Read a resource of the MCP server and return its text. Available resources:"]
    lines += [f"- {item.uri}: {item.description or item.title or item.name}" for item in resources]
    if templates:
        lines.append("Resource templates, with the parts in braces filled in:")
        lines += [f"- {item.uri_template}: {item.description or item.title or item.name}" for item in templates]
    parameters = {
        "type": "object",
        "properties": {"uri": {"type": "string", "description": "URI of the resource to read"}},
        "required": ["uri"],
    }
    return _function(RESOURCE_TOOL, "\n".join(lines), parameters)


def _resource_text(contents: Any) -> str:
    text = getattr(contents, "text", None)
    return text if text is not None else f"[binary resource {contents.uri}, {contents.mime_type}]"


def _python_function(directory: Path) -> dict[str, Any]:
    description = (
        "Run Python code in a new process and return its exit code, standard output and standard error. "
        "numpy, pandas, scipy and scikit-learn are installed. The working directory is "
        f"{directory}; files written there stay for later calls. The process stops after "
        f"{PYTHON_TIMEOUT_SECONDS} seconds."
    )
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "the complete Python program to run"}},
        "required": ["code"],
    }
    return _function(PYTHON_TOOL, description, parameters)


def _run_python(directory: Path, code: str) -> tuple[str, bool]:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"run_{len(list(directory.glob('run_*.py'))) + 1}.py"
    script.write_text(code, encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if key != API_KEY_VARIABLE}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = PYTHON_THREADS
    try:
        done = subprocess.run(
            [sys.executable, str(script)], cwd=directory, env=environment, capture_output=True, text=True,
            timeout=PYTHON_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: the process was stopped after {PYTHON_TIMEOUT_SECONDS} seconds.", True
    text = f"exit code {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    if len(text) > PYTHON_OUTPUT_CHARACTERS:
        half = PYTHON_OUTPUT_CHARACTERS // 2
        text = f"{text[:half]}\n[... {len(text) - PYTHON_OUTPUT_CHARACTERS} characters omitted ...]\n{text[-half:]}"
    return text, done.returncode != 0


def _content_text(blocks: Sequence[Any]) -> str:
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "resource":
            parts.append(_resource_text(block.resource))
        elif block.type == "resource_link":
            parts.append(f"[resource link {block.uri}]")
        else:
            parts.append(f"[{block.type} content omitted]")
    return "\n".join(parts)


def _system_prompt(instructions: str | None, python: bool) -> str:
    prompt = SYSTEM_PROMPT + (PYTHON_PROMPT if python else "")
    if not instructions:
        return prompt
    return f"{prompt}\n\n{INSTRUCTIONS_HEADER}\n{instructions}"


async def _execute(
    client: Client, resources: bool, python_dir: Path | None, name: str, raw: Any
) -> tuple[Any, str, bool]:
    """Runs one requested call and returns the parsed arguments, the text handed back to the model, and the error flag."""
    try:
        arguments = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        return raw, f"Error: the arguments are not valid JSON ({error}).", True
    if not isinstance(arguments, dict):
        return arguments, "Error: the arguments must be a JSON object.", True
    if python_dir is not None and name == PYTHON_TOOL:
        code = arguments.get("code")
        if not isinstance(code, str):
            return arguments, "Error: code must be a string.", True
        text, failed = await asyncio.to_thread(_run_python, python_dir, code)
        return arguments, text, failed
    try:
        if resources and name == RESOURCE_TOOL:
            read = await client.read_resource(str(arguments.get("uri", "")))
            return arguments, "\n".join(_resource_text(item) for item in read.contents), False
        result = await client.call_tool(name, arguments)
    except MCPError as error:
        return arguments, f"Error: {error.message}", True
    text = _content_text(result.content)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return arguments, text, result.is_error


def _summary(steps: list[dict[str, Any]], started: float, answer: str | None, stop_reason: str) -> dict[str, Any]:
    tools = [call for step in steps for call in step["tool_results"]]
    usage = [step["usage"] or {} for step in steps]
    return {
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - started, 3),
        "model_seconds": round(sum(step["seconds"] for step in steps), 3),
        "tool_seconds": round(sum(call["seconds"] for call in tools), 3),
        "model_calls": len(steps),
        "tool_calls": len(tools),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in usage),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        "cache_n": sum((step["timings"] or {}).get("cache_n", 0) for step in steps),
        "prompt_n": sum((step["timings"] or {}).get("prompt_n", 0) for step in steps),
        "final_answer": answer,
        "stop_reason": stop_reason,
    }


def _cause(error: BaseException) -> BaseException:
    # The MCP client runs in anyio task groups, which wrap a failure in an exception group.
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _write(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def run(arguments: argparse.Namespace) -> int:
    api_key = os.environ.get(API_KEY_VARIABLE)
    url = f"{_api_base(arguments.base_url)}/chat/completions"
    command = shlex.split(arguments.server_command)
    # The server inherits this environment except the model API key.
    environment = {key: value for key, value in os.environ.items() if key != API_KEY_VARIABLE}
    server = StdioServerParameters(command=command[0], args=command[1:], env=environment)
    prompt_arguments = dict(arguments.arg)
    log: dict[str, Any] = {
        "model": arguments.model,
        "sampling": SAMPLING,
        "mcp_version": version("mcp"),
        "server_command": arguments.server_command,
        "task": arguments.task,
        "prompt": {"name": arguments.prompt, "arguments": prompt_arguments} if arguments.prompt else None,
        "max_steps": arguments.max_steps,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    answer: str | None = None
    stop_reason = "interrupted"
    try:
        async with Client(server) as client:
            info = client.server_info
            log["server"] = {
                "name": info.name if info else None,
                "version": info.version if info else None,
                "protocol_version": client.protocol_version,
                "instructions": client.instructions,
            }
            listed = await _list_all(client.list_tools, "tools")
            tools = [_function(tool.name, tool.description or "", tool.input_schema) for tool in listed]
            capabilities = client.server_capabilities
            resources: list[Any] = []
            templates: list[Any] = []
            if capabilities.resources is not None:
                resources = await _list_all(client.list_resources, "resources")
                templates = await _list_all(client.list_resource_templates, "resource_templates")
            bridge = bool(resources or templates)
            if bridge:
                tools.append(_resource_tool(resources, templates))
            if arguments.python_dir is not None:
                tools.append(_python_function(arguments.python_dir))
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _system_prompt(client.instructions, arguments.python_dir is not None)}
            ]
            if arguments.prompt:
                prompt = await client.get_prompt(arguments.prompt, prompt_arguments)
                messages += [{"role": item.role, "content": _content_text([item.content])} for item in prompt.messages]
            else:
                messages.append({"role": "user", "content": arguments.task})
            log["tools"] = [tool["function"]["name"] for tool in tools]
            log["resources"] = [item.uri for item in resources] + [item.uri_template for item in templates]
            log["initial_messages"] = [dict(message) for message in messages]
            log["steps"] = steps
            body: dict[str, Any] = {"model": arguments.model, "stream": False, **SAMPLING}
            if tools:
                body.update(tools=tools, tool_choice="auto")

            for index in range(1, arguments.max_steps + 1):
                body["messages"] = messages
                tick = time.perf_counter()
                response = await asyncio.to_thread(_post, url, body, api_key)
                choice = response["choices"][0]
                message = choice["message"]
                calls = message.get("tool_calls") or []
                for number, call in enumerate(calls):
                    call.setdefault("id", f"call_{index}_{number}")
                step: dict[str, Any] = {
                    "step": index,
                    "seconds": round(time.perf_counter() - tick, 3),
                    "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
                    "content": message.get("content"),
                    "tool_calls": calls,
                    "tool_results": [],
                }
                steps.append(step)
                if not calls:
                    answer = message.get("content")
                    stop_reason = "answer" if step["finish_reason"] == "stop" else f"finish_reason {step['finish_reason']}"
                elif index == arguments.max_steps:
                    stop_reason = f"max_steps {arguments.max_steps} reached with tool calls pending"
                else:
                    # The reasoning is dropped from the history, as Qwen's chat template expects.
                    messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
                    for call in calls:
                        name = call["function"]["name"]
                        tick = time.perf_counter()
                        parsed, text, is_error = await _execute(
                            client, bridge, arguments.python_dir, name, call["function"].get("arguments")
                        )
                        step["tool_results"].append({
                            "name": name,
                            "arguments": parsed,
                            "seconds": round(time.perf_counter() - tick, 3),
                            "is_error": is_error,
                            "result": text,
                        })
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                totals = _summary(steps, started, answer, stop_reason)
                called = ", ".join(call["function"]["name"] for call in calls) or "final answer"
                print(
                    f"step {index} | {called} | {totals['total_seconds']:.1f} s | "
                    f"prompt {totals['prompt_tokens']} completion {totals['completion_tokens']} "
                    f"cached {totals['cache_n']} tokens",
                    flush=True,
                )
                _write(arguments.log, log)
                if not calls:
                    break
    except Exception as error:
        cause = _cause(error)
        stop_reason = f"error: {type(cause).__name__}: {cause}"
        raise
    finally:
        log["summary"] = _summary(steps, started, answer, stop_reason)
        _write(arguments.log, log)
    if answer is not None:
        print(answer)
    return 0 if stop_reason == "answer" else 1


def _key_value(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"{text!r} is not of the form key=value")
    return key, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive an MCP server with a model behind an OpenAI-compatible endpoint.")
    parser.add_argument("--server-command", required=True, help="command that starts the MCP server on stdio")
    parser.add_argument("--base-url", required=True, help="endpoint base URL; /v1 is appended when missing")
    parser.add_argument("--model", required=True, help="model name sent to the endpoint")
    opening = parser.add_mutually_exclusive_group(required=True)
    opening.add_argument("--task", help="text of the user request")
    opening.add_argument("--prompt", help="name of a server prompt that opens the conversation")
    parser.add_argument("--arg", type=_key_value, action="append", default=[], help="prompt argument key=value")
    parser.add_argument("--log", type=Path, required=True, help="path of the JSON log")
    parser.add_argument("--max-steps", type=int, default=30, help="maximum number of model calls")
    parser.add_argument("--python-dir", type=Path, default=None, help="working directory of the run_python function; offers it to the model")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.arg and not arguments.prompt:
        parser.error("--arg is only used with --prompt")
    return asyncio.run(run(arguments))


if __name__ == "__main__":
    sys.exit(main())
