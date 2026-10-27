from openai import OpenAI, OpenAIError

from config import API_KEY, BASE_URL, MODEL, SYSTEM_PROMPT
from token_utils import print_usage


# Create one reusable API client.
client_options = {"api_key": API_KEY}

if BASE_URL:
    client_options["base_url"] = BASE_URL

client = OpenAI(**client_options)


def chat_completion(messages, stream=False, temperature=None):
    """
    Send a conversation using the Chat Completions API.

    Returns:
        (answer, usage)

    If the API request fails, returns (None, None).
    """
    request_options = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }

    # Only pass temperature when explicitly requested.
    if temperature is not None:
        request_options["temperature"] = temperature

    try:
        if not stream:
            response = client.chat.completions.create(
                **request_options
            )

            answer = response.choices[0].message.content or ""
            return answer, response.usage

        # Ask for usage in the final streaming chunk when supported.
        request_options["stream_options"] = {"include_usage": True}

        stream_response = client.chat.completions.create(
            **request_options
        )

        answer_parts = []
        usage = None

        print("\nAI: ", end="", flush=True)

        for chunk in stream_response:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

            # The final usage chunk may not contain any choices.
            if not chunk.choices:
                continue

            piece = chunk.choices[0].delta.content

            if piece:
                print(piece, end="", flush=True)
                answer_parts.append(piece)

        print()

        return "".join(answer_parts), usage

    except OpenAIError as error:
        print(f"\nChat Completions API error: {error}")
        return None, None


def responses_completion(user_message, previous_response_id=None):
    """
    Send a request using the Responses API.

    previous_response_id can link a new request to a previous response
    if the configured provider supports this feature.

    Returns:
        (answer, response_id, usage)

    On failure, returns (None, previous_response_id, None).
    """
    request_options = {
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": user_message,
    }

    if previous_response_id:
        request_options["previous_response_id"] = previous_response_id

    try:
        response = client.responses.create(**request_options)

        return (
            response.output_text or "",
            response.id,
            getattr(response, "usage", None),
        )

    except OpenAIError as error:
        print(
            "\nResponses API error. Your provider may not support "
            "the Responses API or previous_response_id.\n"
            f"Details: {error}"
        )
        return None, previous_response_id, None
