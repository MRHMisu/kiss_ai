# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Coding Agent using the Claude Agent SDK.

This module provides a coding agent that uses the Claude Agent SDK to generate
tested Python programs. The agent can use various built-in tools (Read, Bash,
WebSearch, etc.) and custom tools like read_project_file.
"""

from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from pydantic import BaseModel, Field

from kiss.core import DEFAULT_CONFIG

# Built-in tools available in Claude Agent SDK
# These can be enabled via the allowed_tools parameter
BUILTIN_TOOLS = {
    "Read": "Read files from the working directory",
    "Write": "Create or overwrite files",
    "Edit": "Make precise string-based edits to files",
    "MultiEdit": "Make multiple precise string-based edits to files",
    "Glob": "Find files by glob pattern (e.g., **/*.py)",
    "Grep": "Search file contents with regex",
    "Bash": "Run shell commands",
    "WebSearch": "Search the web for information",
    "WebFetch": "Fetch and process content from a URL",
}


# System prompt for generating robust, tested code
SYSTEMS_PROMPT = """You are an expert Python programmer who writes clean, simple, \
and robust code.

## Code Style Guidelines
- Write simple, readable code with minimal indirection
- Avoid unnecessary object attributes and local variables
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Use clear, descriptive names

## Testing Requirements
- Generate comprehensive tests for EVERY function and feature
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Test with real inputs and verify real outputs
- Test edge cases: empty inputs, None values, boundary conditions
- Test error conditions with actual invalid inputs
- Each test should be independent and verify actual behavior

## Code Structure
- Main implementation code first
- Test code in a separate section using unittest or pytest
- Include a __main__ block to run tests

## Available Tools
You have access to the following tools to help with your task:
- read_project_file: Read files from the project directory
- WebSearch: Search the web for documentation, examples, or solutions
- WebFetch: Fetch content from a specific URL
- Read: Read files from the working directory
- Glob: Find files matching a pattern
- Grep: Search file contents

Use these tools when you need to:
- Look up API documentation or library usage
- Find examples of similar implementations
- Understand existing code in the project

## Output Format
Return a dict of the form by carefully and rigorously introspecting on your work.
```python
{
    "status": bool,
    "summary": str,
    "insights": str
}
```
"""


class TaskResult(BaseModel):
    status: bool = Field(
        description=(
            "True if the agent successfully completed the task. "
            "Please introspect on your work to generate the status."
        )
    )
    summary: str = Field(
        description=(
            "A summary of the code generation and modification process. "
            "Please introspect on your work to generate the summary."
        )
    )
    insights: str = Field(
        description=(
            "Actionable insights/instructions to improve future coding. "
            "The insights and instructions MUST be task agnostic, generic, concise "
            "and to the point. "
            "You MUST not generate any task specific insights or instructions "
            "because then the coding agent will not be able to generalize to new "
            "tasks. "
            "Please introspect on your work to generate the insights and "
            "instructions ONLY if you have failed some tool calls."
        )
    )

class ClaudeCodingAgent:
    def __init__(
        self,
        model_name: str,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        base_dir: str = str(Path(DEFAULT_CONFIG.agent.artifact_dir).resolve() / "claude_workdir")
    ):
        if readable_paths is None:
            readable_paths = []
        if writable_paths is None:
            writable_paths = []
        self.base_dir = base_dir
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.readable_paths = {Path(p).resolve() for p in readable_paths}
        self.writable_paths = {Path(p).resolve() for p in writable_paths}

    def _is_subpath(self, target: Path, whitelist: set[Path]) -> bool:
        """Checks if the target path is or is inside any of the whitelisted paths."""
        return any(target == p or p in target.parents for p in whitelist)

    async def permission_handler(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Enforces Read/Write restrictions.

        Args:
            tool_name: The name of the tool being requested.
            tool_input: The input parameters for the tool.
            context: Additional context for the permission request.

        Returns:
            PermissionResultAllow or PermissionResultDeny based on path restrictions.
        """
        path_str = tool_input.get("file_path") or tool_input.get("path")

        if not path_str:
            return PermissionResultAllow(behavior="allow")

        target_path = Path(path_str).resolve()

        # Enforce Read Restrictions
        if tool_name in ["Read", "Grep", "Glob"]:
            if len(self.readable_paths) == 0 or self._is_subpath(
                target_path, self.readable_paths
            ):
                return PermissionResultAllow(behavior="allow")
            msg = f"Access Denied: {path_str} is not in readable whitelist."
            return PermissionResultDeny(behavior="deny", message=msg)

        # Enforce Write Restrictions
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            if len(self.writable_paths) == 0 or self._is_subpath(
                target_path, self.writable_paths
            ):
                return PermissionResultAllow(behavior="allow")
            msg = f"Access Denied: {path_str} is not in writable whitelist."
            return PermissionResultDeny(behavior="deny", message=msg)

        return PermissionResultAllow(behavior="allow")

    async def _prompt_stream(self, task: str) -> Any:
        """Wrap the task prompt as an async iterable for streaming mode.

        The can_use_tool callback requires streaming mode, which needs the prompt
        to be provided as an AsyncIterable with proper message structure.
        """
        yield {
            "type": "user",
            "message": {"role": "user", "content": task}
        }

    async def run(self, task: str) -> dict[str, object] | None:
        options = ClaudeAgentOptions(
            model=self.model_name,
            system_prompt=SYSTEMS_PROMPT,
            output_format=TaskResult.model_json_schema(),
            can_use_tool=self.permission_handler,
            permission_mode="default",  # Use default mode so can_use_tool callback is invoked
            allowed_tools=list(BUILTIN_TOOLS.keys()),
            cwd=str(self.base_dir)
        )

        final_result: dict[str, object] | None = None
        # Use the standalone query() function with streaming prompt for can_use_tool support
        async for message in query(prompt=self._prompt_stream(task), options=options):
            # Handle AssistantMessage which contains ToolUseBlock and TextBlock
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        args_str = ", ".join(
                            f"{k}={repr(v)[:50]}" for k, v in block.input.items()
                        )
                        print(f"[TOOL] {block.name}({args_str})")

                    elif isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

            # Handle UserMessage which contains ToolResultBlock
            elif isinstance(message, UserMessage):
                for content_block in message.content:
                    if isinstance(content_block, ToolResultBlock):
                        content = content_block.content
                        if isinstance(content, str):
                            if len(content) > 200:
                                display = content[:100] + "..." + content[-100:]
                            else:
                                display = content
                            display = display.replace("\n", "\\n")
                        else:
                            content_str = str(content)
                            display = content_str[:100] + "..." + content_str[-100:]
                        status = "ERROR" if content_block.is_error else "OK"
                        print(f"  -> [{status}] {display}")

            elif isinstance(message, ResultMessage):
                # Try structured_output first, fall back to parsing result
                if message.structured_output is not None:
                    final_result = message.structured_output  # type: ignore[assignment]
                elif message.result:
                    # Try to extract JSON from result text
                    final_result = self._parse_result_json(message.result)

        return final_result

    def _parse_result_json(self, result: str) -> dict[str, object] | None:
        """Parse JSON from result text, handling markdown code blocks."""
        import json
        import re

        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())  # type: ignore[return-value, no-any-return]
            except json.JSONDecodeError:
                pass

        # Try to parse as raw JSON
        try:
            return json.loads(result.strip())  # type: ignore[return-value, no-any-return]
        except json.JSONDecodeError:
            pass

        # Return a basic result if we can't parse JSON
        return {"status": True, "summary": result[:500], "insights": ""}

async def main() -> None:
    project_root = Path(DEFAULT_CONFIG.agent.artifact_dir).resolve()
    agent = ClaudeCodingAgent(
        model_name="claude-sonnet-4-5",
        readable_paths=["workdir"],
        writable_paths=["workdir"],
        base_dir=str(project_root)
    )

    task_description = """
    can you write, test, and optimize a fibonacci function in Python that is efficient and correct?
    """
    result = await agent.run(task_description)

    if result:
        print("\n--- FINAL AGENT REPORT ---")
        print(f"SUCCESS: {result['status']}")
        print(f"SUMMARY: {result['summary']}")
        print(f"INSIGHTS: {result['insights']}")

if __name__ == "__main__":
    anyio.run(main)



