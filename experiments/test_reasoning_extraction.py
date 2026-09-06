from data.gsm8k_loader import GSM8KLoader
from data.reasoning_extractor import ReasoningExtractor


def test_split(loader, extractor, split, number_of_examples=3):
    """
    Test reasoning extraction on a few examples
    from a GSM8K dataset split.
    """

    dataset = loader.get_split(split)

    print("\n" + "=" * 70)
    print(f"TESTING SPLIT: {split.upper()}")
    print("=" * 70)

    for index in range(min(number_of_examples, len(dataset))):

        example = dataset[index]

        result = extractor.extract_example(example)

        print(f"\nExample {index}")
        print("-" * 70)

        print("Question:")
        print(result["question"])

        print("\nReasoning Steps:")

        for step in result["reasoning_steps"]:
            print(
                f"  S{step['step_id']}: {step['text']}"
            )

        print("\nFinal Answer:")
        print(result["final_answer"])


def main():
    """
    Test the reasoning extractor against
    real GSM8K train, validation, and test data.
    """

    loader = GSM8KLoader(
        validation_size=0.1,
        seed=42
    )

    loader.load()

    extractor = ReasoningExtractor()

    test_split(
        loader,
        extractor,
        "train"
    )

    test_split(
        loader,
        extractor,
        "validation"
    )

    test_split(
        loader,
        extractor,
        "test"
    )

    print("\n" + "=" * 70)
    print("REASONING EXTRACTION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()