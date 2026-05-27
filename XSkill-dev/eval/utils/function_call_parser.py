"""
Function call parser for OpenAI-compatible API responses.
Parses tool calls and answer tags from model outputs.
"""

import re
import json
import ast
from typing import Tuple, Any, Dict, Union


def _parse_tool_arguments(arguments: Any) -> Tuple[bool, Any]:
    if isinstance(arguments, dict):
        return True, arguments
    if arguments is None:
        return True, {}
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return True, {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return False, f"Invalid JSON in arguments: {arguments}"
        return True, parsed if isinstance(parsed, dict) else {"input": parsed}
    return True, {"input": arguments}


def _parse_tool_payload(payload: str) -> Tuple[str, Any]:
    text = payload.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except Exception:
            return "error", f"Invalid JSON in tool call: {payload}"

    if not isinstance(data, dict):
        return "error", f"Tool call payload must be a JSON object: {payload}"

    tool_name = data.get("name") or data.get("tool_name") or data.get("tool")
    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("parameters")
    if arguments is None:
        arguments = data.get("args")
    if arguments is None:
        arguments = {
            key: value
            for key, value in data.items()
            if key not in {"name", "tool_name", "tool"}
        }

    ok, parsed_arguments = _parse_tool_arguments(arguments)
    if not ok:
        return "error", parsed_arguments
    if not tool_name:
        return "error", "Tool call missing 'name' field"

    return "tool_call", {
        "tool_name": str(tool_name),
        "parameters": parsed_arguments,
    }


def parse_function_call_response(response: Union[str, Dict], text_content: str = None) -> Tuple[str, Any]:
    """
    Parse API response supporting OpenAI function calling and text responses.
    
    Args:
        response: API response content, can be:
            - dict: Response object containing tool_calls
            - str: Plain text response
        text_content: Optional text content (used when response is dict)
        
    Returns:
        (action_type, data) tuple:
        - action_type: "function_call", "answer", "text", "error"
        - data: Parsed data
        
    Examples:
        >>> # OpenAI format function call
        >>> response = {"tool_calls": [{"function": {"name": "web_search", "arguments": '{"query": "test"}'}}]}
        >>> parse_function_call_response(response)
        ("function_call", {"tool_name": "web_search", "parameters": {"query": "test"}})
        
        >>> # Text response with <answer> tag
        >>> response = "Let me think... <answer>42</answer>"
        >>> parse_function_call_response(response)
        ("answer", "42")
    
    Reference:
        https://platform.openai.com/docs/guides/function-calling
    """
    
    # Case 1: Response is dict (may contain function calls)
    if isinstance(response, dict):
        # Check OpenAI format tool_calls
        if "tool_calls" in response and response["tool_calls"]:
            tool_calls = response["tool_calls"]
            
            # Only take the first tool call
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                if len(tool_calls) > 1:
                    print("[Function Call Parser] Warning: Multiple tool calls detected, using only the first one")
                
                first_call = tool_calls[0]
                function_data = first_call.get("function", {})
                
                tool_name = function_data.get("name", "")
                # OpenAI uses JSON string for arguments
                arguments = function_data.get("arguments", "{}")
                
                # Parse arguments (may be string or dict)
                ok, parameters = _parse_tool_arguments(arguments)
                if not ok:
                    return "error", parameters
                
                if not tool_name:
                    return "error", "Tool call missing 'name' field"
                
                return "function_call", {
                    "tool_name": tool_name,
                    "parameters": parameters
                }
        
        # If no function call, check for text content
        if text_content:
            response = text_content
        elif "content" in response:
            response = response["content"]
        elif "text" in response:
            response = response["text"]
        else:
            # No parseable content
            return "text", ""
    
    # Case 2: Response is string
    if isinstance(response, str):
        # Check for text-form tool call. Some OpenAI-compatible local models
        # (including Qwen-family vLLM deployments) may emit the tool call as
        # text instead of returning an OpenAI `tool_calls` field.
        for pattern in (
            r"<tool_call>(.*?)</tool_call>",
            r"<tool>(.*?)</tool>",
            r"```(?:json)?\s*(\{.*?\"(?:name|tool_name|tool)\".*?\})\s*```",
        ):
            match = re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return _parse_tool_payload(match.group(1))

        # Check for <answer> tag (for final answer)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        
        # Plain text response
        return "text", response
    
    # Unknown format
    return "error", f"Unknown response format: {type(response)}"
