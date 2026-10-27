from unittest.mock import patch

from conversation import Conversation
from config import SYSTEM_PROMPT


def test_system_prompt_is_preserved():
    """The system prompt must remain at index zero."""
    conversation = Conversation(max_history_tokens=1000)

    conversation.messages.append({
        "role": "user",
        "content": "Hello",
    })

    conversation.trim_history()

    assert conversation.messages[0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }


def test_old_complete_exchange_is_trimmed():
    """Old user/assistant exchanges should be removed together."""
    conversation = Conversation(max_history_tokens=30)

    conversation.messages.extend([
        {"role": "user", "content": "old question " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "user", "content": "new question"},
    ])

    conversation.trim_history()

    # System prompt remains first.
    assert conversation.messages[0]["role"] == "system"

    # The newest user message remains.
    assert conversation.messages[-1] == {
        "role": "user",
        "content": "new question",
    }


@patch("conversation.chat_completion")
def test_successful_message_is_saved(mock_completion):
    """Both the user message and assistant reply should be stored."""
    mock_completion.return_value = ("Hello there!", None)

    conversation = Conversation(max_history_tokens=1000)
    answer = conversation.send_message("Hello")

    assert answer == "Hello there!"
    assert conversation.messages[-2] == {
        "role": "user",
        "content": "Hello",
    }
    assert conversation.messages[-1] == {
        "role": "assistant",
        "content": "Hello there!",
    }


@patch("conversation.chat_completion")
def test_failed_request_does_not_leave_orphaned_message(mock_completion):
    """A failed API request should roll back the current turn."""
    mock_completion.return_value = (None, None)

    conversation = Conversation(max_history_tokens=1000)
    original_messages = conversation.messages.copy()

    answer = conversation.send_message("Hello")

    assert answer is None
    assert conversation.messages == original_messages
