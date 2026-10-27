import tiktoken

from config import MODEL


# Try to select a tokenizer matching the configured model.
try:
    encoding = tiktoken.encoding_for_model(MODEL)
except KeyError:
    # Fallback if the model name is not recognized by tiktoken.
    encoding = tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text):
    """Return the approximate number of tokens in a text string."""
    return len(encoding.encode(text))


def count_tokens(messages):
    """
    Estimate tokens in a list of chat messages.

    This counts text content only. It does not include all message
    overhead, tool definitions, or provider-specific token accounting.
    """
    total = 0

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            total += count_text_tokens(content)

    return total


def print_usage(usage):
    """Print API-reported input and output token counts when available."""
    if usage is None:
        print("[API token usage unavailable]")
        return

    # Chat Completions uses prompt_tokens/completion_tokens.
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)

    # Responses API uses input_tokens/output_tokens.
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)

    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)

    if input_tokens is not None and output_tokens is not None:
        print(
            f"[API usage: {input_tokens} input tokens, "
            f"{output_tokens} output tokens]"
        )
    else:
        print("[API returned usage in an unrecognized format]")
