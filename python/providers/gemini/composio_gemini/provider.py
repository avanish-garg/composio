"""Gemini provider for Composio SDK.

Returns Python callables compatible with google-genai's Automatic Function
Calling (AFC). The SDK can introspect the callable's signature to derive
FunctionDeclaration schemas and auto-execute tool calls in the chat loop.
"""

import functools
import types as pytypes
import typing as t
import warnings
from inspect import Parameter, Signature

from pydantic import BaseModel

from composio.client.types import Tool
from composio.core.provider import AgenticProvider
from composio.core.provider.agentic import AgenticProviderExecuteFn
from composio.utils.shared import (
    ToolSchemaAliases,
    alias_tool_input_schema,
    get_pydantic_signature_format_from_schema_params,
    json_schema_to_model,
    normalize_tool_arguments,
    validate_and_serialize_tool_arguments,
)

# google-genai is only needed for handle_response (backward compat)
try:
    from google.genai import types as genai_types

    HAS_GENAI = True
except ImportError:
    genai_types = None  # type: ignore
    HAS_GENAI = False


# What ``wrap_tool`` keeps for ``handle_response``: the executor bound to the
# ``tools.get()`` / ``session.tools()`` call that produced the callable, plus the
# schema pieces needed to validate the model's arguments the way AFC does.
_Executor = t.Tuple[AgenticProviderExecuteFn, ToolSchemaAliases, t.Type[BaseModel]]

# Attribute that carries a callable's own ``_Executor``, so ``handle_response``
# can run a call through the exact tools that were offered to the model.
_EXECUTOR_ATTR = "__composio_executor__"

# What ``Tools`` binds into every execute function it hands to ``wrap_tool``.
_CALLER_KEYS = ("user_id", "modifiers")


def _same_executor(a: t.Any, b: t.Any) -> bool:
    """Whether two execute functions would run a tool for the same caller.

    ``Tools`` binds ``user_id`` and ``modifiers`` into a new ``functools.partial``
    on every ``get()``, so compare those instead of identity: the same user
    fetching tools again is not a different caller. Anything else (for example
    the Tool Router session executor) is compared by equality.
    """
    if isinstance(a, functools.partial) and isinstance(b, functools.partial):
        return all(a.keywords.get(key) == b.keywords.get(key) for key in _CALLER_KEYS)
    return bool(a == b)


def _to_serializable(value: t.Any) -> t.Any:
    """Recursively convert Pydantic models (and other non-JSON types) to plain dicts/lists.

    The google-genai SDK's AFC pipeline calls ``convert_if_exist_pydantic_model``
    on function arguments, turning nested dicts into dynamically-generated
    Pydantic ``GeneratedModel`` instances.  These are not JSON-serializable, so
    the Composio ``execute_tool`` call fails.  This helper normalises them back
    to plain Python primitives before handing off to the API.
    """
    # Only fields the model returned: AFC fills omitted optional fields with
    # ``None``, which the tool schema may not allow. Aliases restore reserved
    # names such as ``from``.
    # Pydantic v2 BaseModel
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_unset=True, by_alias=True)
    # Pydantic v1 BaseModel
    if hasattr(value, "dict") and hasattr(value, "__fields__"):
        return value.dict(exclude_unset=True, by_alias=True)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _literal_type(schema: t.Any) -> type:
    """The Python type shared by a typeless property's ``const``/``enum``
    values, or ``object`` when there is none."""
    if not isinstance(schema, dict):
        return object
    if "const" in schema:
        values = [schema["const"]]
    elif isinstance(schema.get("enum"), list) and schema["enum"]:
        values = schema["enum"]
    else:
        return object
    # ``bool`` first: it is a subclass of ``int``.
    for literal_type in (bool, str, int, float):
        if all(type(value) is literal_type for value in values):
            return literal_type
    return object


def _afc_annotation(annotation: t.Any, schema: t.Any = None) -> t.Any:
    """Return an annotation google-genai AFC can convert arguments with.

    AFC reads the callable's signature both to declare the function and to
    convert the arguments the model returns, and the conversion calls
    ``isinstance`` with the annotation. That raises on ``Annotated[...]``,
    parameterized ``Dict`` and ``typing.Any``, so any tool using them failed
    before it ran. The plain annotations declare the same function; the source
    schema is still enforced by ``args_schema``.

    A property without a ``type`` is ``typing.Any``. A bare ``const`` or
    ``enum`` is declared with the type of its values, so ``{"enum": ["asc",
    "desc"]}`` stays a string as it was before the signature carried
    validators. Anything else becomes ``object``, which declares no type and
    lets the conversion accept any value.
    """
    origin = t.get_origin(annotation)
    args = t.get_args(annotation)
    if origin is t.Annotated:
        return _afc_annotation(args[0], schema)
    if origin is list and args:
        return t.List[_afc_annotation(args[0])]  # type: ignore[misc]
    if origin is dict:
        return dict
    if origin in (t.Union, pytypes.UnionType):
        return t.Union[tuple(_afc_annotation(arg) for arg in args)]
    if annotation is t.Any:
        return _literal_type(schema)
    return annotation


def _process_execution_result(result: t.Any) -> t.Dict:
    """Process a tool execution result into a dict suitable for Gemini function responses."""
    if not isinstance(result, dict):
        return {"result": result}

    if result.get("successful", True) and "data" in result:
        data = result["data"]
        return data if isinstance(data, dict) else {"result": data}

    if not result.get("successful", True):
        return {
            "error": result.get("error", "Tool execution failed"),
            "details": result,
        }

    return result


class GeminiProvider(AgenticProvider[t.Callable, list[t.Callable]], name="gemini"):
    """Composio toolset for Google AI Python Gemini framework.

    Returns Python callables compatible with google-genai's Automatic Function
    Calling (AFC). Pass the result of ``wrap_tools()`` directly to
    ``GenerateContentConfig(tools=...)`` and the SDK will auto-execute tool
    calls in the ``chat.send_message()`` loop.
    """

    __schema_skip_defaults__ = True

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        # Last executor wrapped for each slug. Only ``handle_response(response)``
        # without ``tools`` reads it, so it cannot tell callers apart.
        self._executors: t.Dict[str, _Executor] = {}
        # Slugs wrapped for more than one caller since, so the registry above is
        # ambiguous for them.
        self._ambiguous_slugs: t.Set[str] = set()

    def wrap_tool(
        self,
        tool: Tool,
        execute_tool: AgenticProviderExecuteFn,
    ) -> t.Callable:
        """Wrap a Composio tool as a Python callable for google-genai AFC.

        The returned function has ``__name__``, ``__doc__``, ``__signature__``
        and ``__annotations__`` set so the google-genai SDK can:

        1. Derive a ``FunctionDeclaration`` schema via ``from_callable()``
        2. Store it in the AFC ``function_map`` for automatic execution
        """
        aliases = alias_tool_input_schema(schema=tool.input_parameters)
        # Defaults stay out of the function declaration (see below), but the
        # validation model needs them: without a default every optional field
        # would be required.
        args_schema = json_schema_to_model(aliases.schema, skip_default=False)
        executor: _Executor = (execute_tool, aliases, args_schema)
        previous = self._executors.get(tool.slug)
        if previous is not None and not _same_executor(previous[0], execute_tool):
            self._ambiguous_slugs.add(tool.slug)
        self._executors[tool.slug] = executor

        def function(**kwargs: t.Any) -> t.Dict:
            """Composio tool execution wrapper."""
            kwargs = _to_serializable(kwargs)
            kwargs = validate_and_serialize_tool_arguments(
                args_schema,
                normalize_tool_arguments(kwargs),
            )
            kwargs = aliases.restore_arguments(kwargs)
            result = execute_tool(tool.slug, kwargs)
            return _process_execution_result(result)

        # Create a real function object (passes inspect.isfunction)
        action_func = pytypes.FunctionType(
            function.__code__,
            globals=globals(),
            name=tool.slug,
            closure=function.__closure__,
        )

        # Build typed signature from JSON schema.
        # Uses get_pydantic_signature_format_from_schema_params (not
        # get_signature_format_from_schema_params) because the pydantic variant
        # goes through json_schema_to_pydantic_type() which produces
        # parameterized generics (e.g. List[str] instead of bare List).
        # The google-genai SDK requires parameterized array types — bare List
        # generates {"type": "ARRAY"} without "items", which the API rejects.
        properties = aliases.schema.get("properties") or {}
        sig_params = [
            param.replace(
                annotation=_afc_annotation(param.annotation, properties.get(param.name))
            )
            for param in get_pydantic_signature_format_from_schema_params(
                schema_params=aliases.schema,
                skip_default=True,
            )
        ]
        action_func.__signature__ = Signature(parameters=sig_params)  # type: ignore
        action_func.__doc__ = tool.description or f"Execute {tool.slug}"

        # Build __annotations__ for typing.get_type_hints() compatibility
        annotations: t.Dict[str, t.Any] = {}
        for param in sig_params:
            if param.annotation is not Parameter.empty:
                annotations[param.name] = param.annotation
        annotations["return"] = dict
        action_func.__annotations__ = annotations
        setattr(action_func, _EXECUTOR_ATTR, executor)

        return action_func

    def wrap_tools(
        self,
        tools: t.Sequence[Tool],
        execute_tool: AgenticProviderExecuteFn,
    ) -> list[t.Callable]:
        """Wrap multiple Composio tools as Python callables for google-genai AFC."""
        return [self.wrap_tool(tool, execute_tool) for tool in tools]

    # --- Backward compatibility: manual function calling ---

    def handle_response(
        self,
        response: t.Any,
        tools: t.Optional[t.Sequence[t.Callable]] = None,
    ) -> tuple[list, bool]:
        """Manually handle function calls in a Gemini response.

        Provided for backward compatibility with code that uses manual function
        calling instead of AFC. For new code, pass the callables from
        ``wrap_tools()`` to ``GenerateContentConfig(tools=...)`` and AFC will
        handle execution automatically.

        A response does not say whose tools produced it. Pass the callables the
        request was made with as ``tools`` and each function call runs as the
        user (or Tool Router session) those tools were fetched for. Without
        ``tools`` the provider falls back to the executor of the most recent
        ``wrap_tools()`` call for each function name, which is the wrong user
        when one ``Composio`` instance fetches tools for several users; that
        case emits a warning.

        Args:
            response: The ``GenerateContentResponse`` to read function calls from.
            tools: The callables returned by ``composio.tools.get(...)`` or
                ``session.tools()`` for this conversation. Function calls that
                match none of them are skipped.

        Returns:
            tuple: ``(function_responses, executed)`` where *function_responses*
            are ``genai_types.Part`` objects ready to send back, and *executed*
            is ``True`` if any functions were executed.
        """
        if not HAS_GENAI:
            return [], False

        if not (hasattr(response, "candidates") and response.candidates):
            return [], False

        candidate = response.candidates[0]
        if not (hasattr(candidate, "content") and candidate.content.parts):
            return [], False

        executors: t.Mapping[str, _Executor]
        if tools is None:
            executors = self._executors
        else:
            executors = {}
            for tool in tools:
                executor = getattr(tool, _EXECUTOR_ATTR, None)
                if executor is not None:
                    executors[tool.__name__] = executor

        function_responses: list = []
        executed = False

        for part in candidate.content.parts:
            if not (hasattr(part, "function_call") and part.function_call):
                continue

            fc = part.function_call
            if fc.name not in executors:
                continue

            if tools is None and fc.name in self._ambiguous_slugs:
                warnings.warn(
                    f"GeminiProvider.handle_response() ran {fc.name!r} with the "
                    "most recent composio.tools.get() / session.tools() call "
                    "that wrapped it, but that tool was wrapped for more than "
                    "one user or session, so this may not be the one that "
                    "produced the response. Pass the tools the response was "
                    "generated with: handle_response(response, tools=tools).",
                    UserWarning,
                    stacklevel=2,
                )

            # Same validate -> serialize -> alias sequence as the AFC callable.
            execute_tool, aliases, args_schema = executors[fc.name]
            arguments = validate_and_serialize_tool_arguments(
                args_schema,
                normalize_tool_arguments(_to_serializable(dict(fc.args))),
            )
            arguments = aliases.restore_arguments(arguments)
            result = execute_tool(slug=fc.name, arguments=arguments)
            processed = _process_execution_result(result)

            function_responses.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=fc.name, response=processed
                    )
                )
            )
            executed = True

        return function_responses, executed
