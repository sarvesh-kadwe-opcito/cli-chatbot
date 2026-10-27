from config import SYSTEM_PROMPT, MAX_HISTORY_TOKENS
from llm_client import chat_completion
from token_utils import count_tokens, print_usage


class Conversation:
    """Manage local chat history and send messages to the LLM."""

    def __init__(self, max_history_tokens=MAX_HISTORY_TOKENS):
        self.max_history_tokens = max_history_tokens

        # The system prompt must always remain at index 0.
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def trim_history(self):
        """
        Remove the oldest complete user/assistant exchanges.

        The system prompt is preserved, as is the newest user message.
        If the newest message alone exceeds the budget, keep it and warn.
        """
        system_message = self.messages[0]
        history = self.messages[1:].copy()

        while count_tokens([system_message] + history) > self.max_history_tokens:
            # Remove the oldest complete exchange together.
            if (
                len(history) >= 2
                and history[0]["role"] == "user"
                and history[1]["role"] == "assistant"
            ):
                del history[:2]
                continue

            # A leading assistant message should not normally occur
            # in valid alternating history. Remove it if it is old.
            if (
                len(history) >= 2
                and history[0]["role"] == "assistant"
            ):
                history.pop(0)
                continue

            # Preserve the newest user message, even if it is too long.
            if len(history) <= 1:
                break

            history.pop(0)

        self.messages = [system_message] + history

    def send_message(self, user_message, stream=False, temperature=None):
        """Send one user message and save the assistant's reply."""

        # Remember the current state so a failed API call can be rolled back.
        previous_messages = self.messages.copy()

        self.messages.append({
            "role": "user",
            "content": user_message,
        })

        # Trim old history before sending the request.
        self.trim_history()

        current_tokens = count_tokens(self.messages)

        if current_tokens > self.max_history_tokens:
            print(
                f"[Warning: approximately {current_tokens} text tokens "
                f"exceed the budget of {self.max_history_tokens}.]"
            )

        answer, usage = chat_completion(
            self.messages,
            stream=stream,
            temperature=temperature,
        )

        # Do not leave a user message without a reply if the request fails.
        if answer is None:
            self.messages = previous_messages
            return None

        self.messages.append({
            "role": "assistant",
            "content": answer,
        })

        print_usage(usage)
        print(
            f"[Approximate local history tokens: "
            f"{count_tokens(self.messages)}]"
        )

        return answer
