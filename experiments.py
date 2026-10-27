import difflib

from config import BASE_DIR, MODEL, SYSTEM_PROMPT
from llm_client import chat_completion
from token_utils import print_usage


def run_temperature_experiment():
    """Compare five runs at temperature 0 and five at temperature 1.2."""

    prompt = (
        "Explain Python list comprehensions in simple language. "
        "Give one short example."
    )

    temperatures = [0.0, 1.2]
    repetitions = 5
    results = {}

    for temperature in temperatures:
        results[temperature] = []

        for run_number in range(1, repetitions + 1):
            print(
                f"\nTemperature={temperature}, "
                f"run {run_number}/{repetitions}"
            )

            # Use the same prompt for each run so the outputs can be compared.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            answer, usage = chat_completion(
                messages,
                temperature=temperature,
            )

            if answer is None:
                answer = "[Request failed; see the error above.]"
            else:
                print(answer)
                print_usage(usage)

            results[temperature].append(answer)

    # Save every result to a text file.
    output_file = BASE_DIR / "temperature_experiment.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write(f"Model: {MODEL}\n")
        file.write(f"Prompt: {prompt}\n")

        for temperature, answers in results.items():
            file.write(f"\n{'=' * 60}\n")
            file.write(f"TEMPERATURE: {temperature}\n")

            for index, answer in enumerate(answers, start=1):
                file.write(f"\n--- Run {index} ---\n")
                file.write(answer + "\n")

        # Compare the first successful-looking output at each temperature.
        # The file includes failed-request placeholders if a call failed.
        file.write("\n\n--- DIFF: FIRST RUN AT EACH TEMPERATURE ---\n")

        file.writelines(
            difflib.unified_diff(
                results[0.0][0].splitlines(keepends=True),
                results[1.2][0].splitlines(keepends=True),
                fromfile="temperature_0.0",
                tofile="temperature_1.2",
            )
        )

    print(f"\nResults saved to: {output_file}")
