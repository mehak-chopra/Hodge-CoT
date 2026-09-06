import re


class ReasoningExtractor:
    """
    Extracts structured reasoning steps from GSM8K solutions.

    Responsibilities
    ----------------
    - Extract the final answer marked by "####".
    - Separate reasoning from the final answer.
    - Extract ordered reasoning steps.
    - Clean reasoning text.
    - Process individual examples or complete dataset splits.

    This class does NOT:
    - classify reasoning step types,
    - create graph relationships,
    - assign edge weights,
    - calculate Hodge components,
    - detect inconsistencies,
    - determine mathematical correctness.
    """

    def __init__(self):
        """
        Initialize the reasoning extractor.
        """

        # GSM8K uses "####" to mark the final answer.
        self.final_answer_pattern = re.compile(
            r"####\s*(.+?)\s*$",
            re.DOTALL
        )

    def extract_final_answer(self, solution):
        """
        Extract the final answer from a GSM8K solution.

        Parameters
        ----------
        solution : str
            Complete GSM8K solution.

        Returns
        -------
        str
            Final answer.
        """

        if not isinstance(solution, str):
            raise TypeError("Solution must be a string.")

        match = self.final_answer_pattern.search(solution)

        if not match:
            raise ValueError(
                "Could not find the GSM8K final answer marker '####'."
            )

        return match.group(1).strip()

    def remove_final_answer(self, solution):
        """
        Remove the final answer section from a GSM8K solution.

        Parameters
        ----------
        solution : str
            Complete GSM8K solution.

        Returns
        -------
        str
            Reasoning portion of the solution.
        """

        if not isinstance(solution, str):
            raise TypeError("Solution must be a string.")

        match = self.final_answer_pattern.search(solution)

        if not match:
            raise ValueError(
                "Could not find the GSM8K final answer marker '####'."
            )

        reasoning = solution[:match.start()]

        return reasoning.strip()

    def clean_step(self, step):
        """
        Clean an individual reasoning step.

        Parameters
        ----------
        step : str
            Raw reasoning step.

        Returns
        -------
        str
            Cleaned reasoning step.
        """

        if not isinstance(step, str):
            raise TypeError("Reasoning step must be a string.")

        # Remove leading/trailing whitespace.
        step = step.strip()

        # Replace repeated whitespace, including line breaks,
        # with a single space.
        step = re.sub(r"\s+", " ", step)

        return step

    def split_into_steps(self, reasoning):
        """
        Split reasoning into individual logical steps.

        GSM8K normally stores each reasoning statement on a
        separate line. However, a long statement may sometimes
        wrap across multiple physical lines.

        Therefore:
        1. Start with non-empty lines.
        2. Join lines that appear to be continuation text.
        3. Preserve the original reasoning order.

        Parameters
        ----------
        reasoning : str
            Reasoning portion of a GSM8K solution.

        Returns
        -------
        list[str]
            Ordered reasoning steps.
        """

        if not isinstance(reasoning, str):
            raise TypeError("Reasoning must be a string.")

        lines = [
            line.strip()
            for line in reasoning.splitlines()
            if line.strip()
        ]

        if not lines:
            return []

        steps = []

        for line in lines:

            # A line beginning with a normal sentence-style
            # continuation should remain part of the previous step.
            #
            # This prevents accidental splitting when a sentence
            # wraps across multiple physical lines.
            if steps and self.is_continuation(line):
                steps[-1] = f"{steps[-1]} {line}"
            else:
                steps.append(line)

        return [
            self.clean_step(step)
            for step in steps
            if self.clean_step(step)
        ]

    def is_continuation(self, line):
        """
        Determine whether a line is likely a continuation
        of the previous reasoning step.

        Parameters
        ----------
        line : str
            Current line.

        Returns
        -------
        bool
            True if the line appears to continue the previous step.
        """

        # Lowercase sentence starters commonly indicate that
        # the line is continuing the previous sentence.
        continuation_starters = (
            "and ",
            "or ",
            "but ",
            "because ",
            "which ",
            "that ",
            "so ",
            "then ",
            "than ",
            "to ",
            "of ",
            "for ",
            "with ",
            "from ",
            "by ",
            "in ",
            "on ",
            "at ",
        )

        lowered = line.lower()

        if lowered.startswith(continuation_starters):
            return True

        # A line beginning with punctuation or a mathematical
        # expression is also likely to be a continuation.
        if line.startswith(("$", "%", "=", "+", "-", "*", "/", ")")):
            return True

        return False

    def extract(self, solution):
        """
        Extract a structured reasoning trace from one GSM8K solution.

        Parameters
        ----------
        solution : str
            Complete GSM8K solution.

        Returns
        -------
        dict
            Dictionary containing:
                - reasoning_steps
                - final_answer
        """

        final_answer = self.extract_final_answer(solution)

        reasoning = self.remove_final_answer(solution)

        raw_steps = self.split_into_steps(reasoning)

        if not raw_steps:
            raise ValueError(
                "No reasoning steps were extracted from the solution."
            )

        reasoning_steps = []

        for index, step in enumerate(raw_steps, start=1):

            reasoning_steps.append(
                {
                    "step_id": index,
                    "text": step
                }
            )

        return {
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer
        }

    def extract_example(self, example):
        """
        Extract reasoning from one GSM8K dataset example.

        Parameters
        ----------
        example : dict
            GSM8K example containing an "answer" field.

        Returns
        -------
        dict
            Structured reasoning information.
        """

        if "answer" not in example:
            raise KeyError(
                "GSM8K example does not contain an 'answer' field."
            )

        result = self.extract(example["answer"])

        # Preserve the original question when available.
        result["question"] = example.get("question")

        return result

    def extract_dataset(self, dataset):
        """
        Extract reasoning from an entire GSM8K dataset split.

        Parameters
        ----------
        dataset : Dataset
            Hugging Face dataset split.

        Returns
        -------
        list[dict]
            Successfully extracted reasoning examples.

        Notes
        -----
        Examples that cannot be extracted are skipped and
        recorded in extraction_errors.
        """

        extracted_examples = []
        extraction_errors = []

        for index, example in enumerate(dataset):

            try:
                result = self.extract_example(example)

                result["dataset_index"] = index

                extracted_examples.append(result)

            except (ValueError, TypeError, KeyError) as error:

                extraction_errors.append(
                    {
                        "dataset_index": index,
                        "error": str(error)
                    }
                )

        return {
            "examples": extracted_examples,
            "errors": extraction_errors
        }


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Test example
    # ---------------------------------------------------------

    example_solution = """
    A kilogram of chicken costs $6 - $2 = $<<6-2=4>>4.
    Three kilograms of chicken cost $4 x 3 = $<<4*3=12>>12.
    So, a 3-kilogram of chicken and a kilogram of pork cost
    $12 + $6 = $<<12+6=18>>18.
    #### 18
    """

    extractor = ReasoningExtractor()

    result = extractor.extract(example_solution)

    print("\nExtracted Reasoning Steps:")

    for step in result["reasoning_steps"]:
        print(
            f"S{step['step_id']}: {step['text']}"
        )

    print("\nFinal Answer:")
    print(result["final_answer"])