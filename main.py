import argparse

from config import MODEL, MAX_HISTORY_TOKENS
from conversation import Conversation
from experiments import run_temperature_experiment
from llm_client import responses_completion
from token_utils import print_usage


def run_chat_mode(args):
    """Run the normal Chat Completions chatbot."""

    # Use a small budget when testing history trimming.
    max_tokens = 80 if args.tiny_budget else args.max_history_tokens

    conversation = Conversation(max_history_tokens=max_tokens)

    print(f"Chatbot started using {MODEL}")
    print(f"Approximate history-token budget: {max_tokens}")
    print(f"Streaming: {'ON' if args.stream else 'OFF'}")
    print("Type /quit to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        answer = conversation.send_message(
            user_input,
            stream=args.stream,
            temperature=args.temperature,
        )

        # Streaming mode prints the answer as it arrives.
        # Non-streaming mode prints it after the full response arrives.
        if answer is not None and not args.stream:
            print(f"\nAI: {answer}")


def run_responses_mode(stateful=False):
    """
    Run an interactive chatbot using the Responses API.

    In stateful mode, pass the previous response ID on each turn.
    This depends on the configured provider supporting that feature.
    """
    previous_response_id = None

    print(f"Responses API mode using {MODEL}")
    print(f"Stateful mode: {'ON' if stateful else 'OFF'}")
    print("Type /quit to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        answer, response_id, usage = responses_completion(
            user_input,
            previous_response_id=(
                previous_response_id if stateful else None
            ),
        )

        if answer is None:
            # Keep the previous ID after a failed request.
            continue

        print(f"\nAI: {answer}")
        print_usage(usage)

        if stateful:
            previous_response_id = response_id


def main():
    """Parse CLI arguments and launch the requested mode."""

    parser = argparse.ArgumentParser(
        description="Modular CLI chatbot learning project"
    )

    parser.add_argument(
        "--mode",
        choices=["chat", "responses", "stateful", "experiment"],
        default="chat",
        help="Select the chatbot mode or temperature experiment.",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream Chat Completions output.",
    )

    parser.add_argument(
        "--tiny-budget",
        action="store_true",
        help="Use an 80-token history budget to test trimming.",
    )

    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=MAX_HISTORY_TOKENS,
        help="Approximate token budget for local chat history.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature setting for Chat Completions.",
    )

    args = parser.parse_args()

    if args.max_history_tokens < 1:
        parser.error("--max-history-tokens must be greater than zero")

    if args.mode == "chat":
        run_chat_mode(args)

    elif args.mode == "responses":
        run_responses_mode(stateful=False)

    elif args.mode == "stateful":
        run_responses_mode(stateful=True)

    elif args.mode == "experiment":
        run_temperature_experiment()


if __name__ == "__main__":
    main()
