import re


class RelationshipBuilder:
    """
    Builds relationships between reasoning steps.

    Responsibilities
    ----------------
    - Detect numerical values in reasoning steps.
    - Detect explicit arithmetic dependencies.
    - Detect sequential dependencies.
    - Detect references to earlier reasoning steps.
    - Assign relationship confidence scores.
    - Preserve evidence explaining why a relationship was created.
    - Produce a consistent relationship representation for the
      reasoning graph and future edge-flow construction.

    Current relationship types
    --------------------------
    1. depends_on
        A later reasoning step uses information from an earlier step.

    2. follows
        A reasoning step directly follows another step in the
        extracted reasoning sequence.

    3. references
        A reasoning step explicitly refers to an earlier step.

    Important
    ---------
    The relationship type is categorical graph information.

    The confidence score is NOT yet the final mathematical Hodge
    edge flow. Edge-flow construction will be handled separately
    after the relationship structure has been validated.

    This class does NOT:
        - perform Hodge decomposition
        - calculate curl
        - calculate harmonic components
        - detect inconsistencies
        - repair reasoning
        - determine final answer correctness
    """

    def __init__(self):
        # Matches integers and decimal numbers, including negatives.
        self.number_pattern = re.compile(
            r"(?<![\w.])-?\d+(?:\.\d+)?"
        )

        # Phrases commonly used when a reasoning step refers to
        # an earlier step or previously calculated result.
        self.reference_patterns = [
            re.compile(
                r"\b(as\s+(?:calculated|mentioned|shown|determined)"
                r")\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(from\s+(?:above|earlier|previously))\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(previous(?:\s+step|\s+calculation)?)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(above\s+(?:calculation|result|value))\b",
                re.IGNORECASE
            ),
        ]

    # ------------------------------------------------------------------
    # Number extraction
    # ------------------------------------------------------------------

    def extract_numbers(self, text):
        """
        Extract numerical values from a reasoning step.

        Returns
        -------
        list[float]
            Numerical values appearing in the text.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Reasoning step text must be a string."
            )

        numbers = self.number_pattern.findall(text)

        return [
            float(number)
            for number in numbers
        ]

    # ------------------------------------------------------------------
    # Arithmetic dependency detection
    # ------------------------------------------------------------------

    def find_arithmetic_dependencies(
        self,
        reasoning_steps
    ):
        """
        Detect arithmetic dependencies between reasoning steps.

        A later step may depend on an earlier step when numerical
        information from the earlier step is reused in the later step.

        The detector records:
            - source step
            - target step
            - relationship type
            - shared numerical values
            - confidence score
            - evidence

        Notes
        -----
        Shared numerical values are treated as evidence of a possible
        dependency, not as mathematical proof of dependency.

        Returns
        -------
        list[dict]
            Detected arithmetic relationships.
        """

        if not reasoning_steps:
            raise ValueError(
                "No reasoning steps were provided."
            )

        relationships = []

        for target_index in range(
            len(reasoning_steps)
        ):
            target_step = reasoning_steps[
                target_index
            ]

            target_numbers = set(
                self.extract_numbers(
                    target_step["text"]
                )
            )

            if not target_numbers:
                continue

            for source_index in range(
                target_index
            ):
                source_step = reasoning_steps[
                    source_index
                ]

                source_numbers = set(
                    self.extract_numbers(
                        source_step["text"]
                    )
                )

                if not source_numbers:
                    continue

                shared_numbers = (
                    source_numbers
                    .intersection(target_numbers)
                )

                if not shared_numbers:
                    continue

                # Calculate how much of the target's numerical
                # information is shared with the source.
                confidence = (
                    len(shared_numbers)
                    / len(target_numbers)
                )

                relationships.append(
                    {
                        "source": source_step[
                            "step_id"
                        ],
                        "target": target_step[
                            "step_id"
                        ],
                        "relation": "depends_on",
                        "confidence": round(
                            confidence,
                            4
                        ),
                        "evidence": {
                            "type": "shared_numeric_values",
                            "shared_values": sorted(
                                shared_numbers
                            )
                        }
                    }
                )

        return relationships

    # ------------------------------------------------------------------
    # Sequential relationships
    # ------------------------------------------------------------------

    def find_sequential_relationships(
        self,
        reasoning_steps
    ):
        """
        Create structural relationships between consecutive
        reasoning steps.

        Example
        -------
        S1 -> S2
        S2 -> S3
        S3 -> S4

        These relationships represent the ordering of the reasoning
        trace. They do not necessarily imply mathematical dependency.
        """

        if not reasoning_steps:
            raise ValueError(
                "No reasoning steps were provided."
            )

        relationships = []

        for source_step, target_step in zip(
            reasoning_steps,
            reasoning_steps[1:]
        ):
            relationships.append(
                {
                    "source": source_step[
                        "step_id"
                    ],
                    "target": target_step[
                        "step_id"
                    ],
                    "relation": "follows",
                    "confidence": 1.0,
                    "evidence": {
                        "type": "sequential_order"
                    }
                }
            )

        return relationships

    # ------------------------------------------------------------------
    # Reference detection
    # ------------------------------------------------------------------

    def contains_reference(self, text):
        """
        Determine whether a reasoning step explicitly refers to
        an earlier calculation or result.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Reasoning step text must be a string."
            )

        for pattern in self.reference_patterns:
            if pattern.search(text):
                return True

        return False

    def find_reference_relationships(
        self,
        reasoning_steps
    ):
        """
        Detect explicit references to earlier reasoning.

        When a step contains language such as:
            - "as calculated above"
            - "from the previous step"
            - "previous calculation"

        it is connected to the immediately preceding reasoning step.

        Returns
        -------
        list[dict]
            Detected reference relationships.
        """

        if not reasoning_steps:
            raise ValueError(
                "No reasoning steps were provided."
            )

        relationships = []

        for index in range(
            1,
            len(reasoning_steps)
        ):
            target_step = reasoning_steps[index]

            if self.contains_reference(
                target_step["text"]
            ):
                source_step = reasoning_steps[
                    index - 1
                ]

                relationships.append(
                    {
                        "source": source_step[
                            "step_id"
                        ],
                        "target": target_step[
                            "step_id"
                        ],
                        "relation": "references",
                        "confidence": 0.9,
                        "evidence": {
                            "type": "explicit_reference"
                        }
                    }
                )

        return relationships

    # ------------------------------------------------------------------
    # Duplicate relationship handling
    # ------------------------------------------------------------------

    def merge_relationships(
        self,
        relationships
    ):
        """
        Merge multiple pieces of evidence describing the same
        source-target relationship.

        If the same pair of steps has multiple relationships,
        their evidence is preserved instead of silently creating
        duplicate edges.
        """

        merged = {}

        for relationship in relationships:
            key = (
                relationship["source"],
                relationship["target"]
            )

            if key not in merged:
                merged[key] = {
                    "source": relationship[
                        "source"
                    ],
                    "target": relationship[
                        "target"
                    ],
                    "relations": [],
                    "confidence": 0.0,
                    "evidence": []
                }

            entry = merged[key]

            relation = relationship[
                "relation"
            ]

            if relation not in entry[
                "relations"
            ]:
                entry["relations"].append(
                    relation
                )

            entry["confidence"] = max(
                entry["confidence"],
                relationship.get(
                    "confidence",
                    0.0
                )
            )

            evidence = relationship.get(
                "evidence"
            )

            if evidence is not None:
                entry["evidence"].append(
                    evidence
                )

        return list(
            merged.values()
        )

    # ------------------------------------------------------------------
    # Main relationship builder
    # ------------------------------------------------------------------

    def build(
        self,
        reasoning_steps,
        include_sequential=True,
        include_references=True,
        include_arithmetic=True
    ):
        """
        Build the complete relationship structure.

        Parameters
        ----------
        reasoning_steps : list[dict]
            Extracted reasoning steps.

        include_sequential : bool
            Whether to include sequential relationships.

        include_references : bool
            Whether to detect explicit references.

        include_arithmetic : bool
            Whether to detect numerical dependencies.

        Returns
        -------
        list[dict]
            Complete relationship structure.
        """

        if not reasoning_steps:
            raise ValueError(
                "No reasoning steps were provided."
            )

        relationships = []

        if include_sequential:
            relationships.extend(
                self.find_sequential_relationships(
                    reasoning_steps
                )
            )

        if include_arithmetic:
            relationships.extend(
                self.find_arithmetic_dependencies(
                    reasoning_steps
                )
            )

        if include_references:
            relationships.extend(
                self.find_reference_relationships(
                    reasoning_steps
                )
            )

        return self.merge_relationships(
            relationships
        )


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    reasoning_steps = [
        {
            "step_id": 1,
            "text": "Alice initially has 10 apples."
        },
        {
            "step_id": 2,
            "text": "Alice gives 3 apples to Bob."
        },
        {
            "step_id": 3,
            "text": (
                "The number of apples remaining "
                "is 10 - 3 = 7."
            )
        },
        {
            "step_id": 4,
            "text": (
                "Therefore, Alice has 7 apples remaining."
            )
        }
    ]

    builder = RelationshipBuilder()

    relationships = builder.build(
        reasoning_steps
    )

    print("\nDetected Relationships:")
    print("-" * 70)

    for relationship in relationships:
        print(
            f"S{relationship['source']} -> "
            f"S{relationship['target']} | "
            f"relations="
            f"{relationship['relations']} | "
            f"confidence="
            f"{relationship['confidence']}"
        )

        print(
            f"  evidence="
            f"{relationship['evidence']}"
        )