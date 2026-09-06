import textwrap

import matplotlib.pyplot as plt
import networkx as nx


class ReasoningGraph:
    """
    Represents a structured reasoning trace as a directed graph.

    Each reasoning step is represented as a node.
    Relationships between reasoning steps are represented as
    directed edges.

    This class is responsible for graph construction and
    structural validation.

    It does NOT perform:
        - Hodge decomposition
        - inconsistency detection
        - reasoning repair
        - evaluation
    """

    def __init__(
        self,
        question,
        final_answer,
        ground_truth=None
    ):
        """
        Initialize a reasoning graph.

        Parameters
        ----------
        question : str
            Original reasoning question.

        final_answer : str
            Answer produced by the reasoning trace.

        ground_truth : str, optional
            Correct answer, when available.
        """

        self.question = question
        self.final_answer = final_answer
        self.ground_truth = ground_truth

        # Directed graph used as the reasoning representation.
        self.graph = nx.DiGraph()

        # Store graph-level metadata.
        self.graph.graph["question"] = question
        self.graph.graph["final_answer"] = final_answer
        self.graph.graph["ground_truth"] = ground_truth

    def add_reasoning_step(
        self,
        step_id,
        text,
        step_type="unknown"
    ):
        """
        Add one reasoning step as a graph node.

        Parameters
        ----------
        step_id : int
            Unique reasoning-step identifier.

        text : str
            Reasoning statement.

        step_type : str
            Semantic type of the step.

            The default is "unknown" because the reasoning
            extractor does not determine semantic step types.
        """

        if step_id in self.graph.nodes:
            raise ValueError(
                f"Reasoning step {step_id} already exists."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Reasoning step {step_id} must contain text."
            )

        self.graph.add_node(
            step_id,
            text=text.strip(),
            step_type=step_type
        )

    def add_relationship(
        self,
        source,
        target,
        relation="follows",
        weight=1.0
    ):
        """
        Add a directed relationship between two reasoning steps.

        Parameters
        ----------
        source : int
            Source reasoning-step ID.

        target : int
            Target reasoning-step ID.

        relation : str
            Type of relationship.

        weight : float
            Structural edge weight.

            The default value of 1.0 is a placeholder for the
            initial graph representation. It is NOT yet the
            mathematically defined Hodge edge flow.
        """

        if source not in self.graph.nodes:
            raise ValueError(
                f"Source reasoning step {source} does not exist."
            )

        if target not in self.graph.nodes:
            raise ValueError(
                f"Target reasoning step {target} does not exist."
            )

        if not isinstance(weight, (int, float)):
            raise TypeError(
                "Relationship weight must be numeric."
            )

        self.graph.add_edge(
            source,
            target,
            relation=relation,
            weight=float(weight)
        )

    def add_extracted_steps(self, reasoning_steps):
        """
        Add reasoning steps produced by ReasoningExtractor.

        Parameters
        ----------
        reasoning_steps : list[dict]
            Structured reasoning steps in the format:

            [
                {
                    "step_id": 1,
                    "text": "..."
                },
                ...
            ]

        Notes
        -----
        Step types are initially set to "unknown".

        Semantic classification belongs to the graph
        reasoning layer and can be added later.
        """

        if not reasoning_steps:
            raise ValueError(
                "No reasoning steps were provided."
            )

        for step in reasoning_steps:

            if "step_id" not in step:
                raise KeyError(
                    "Reasoning step is missing 'step_id'."
                )

            if "text" not in step:
                raise KeyError(
                    "Reasoning step is missing 'text'."
                )

            self.add_reasoning_step(
                step_id=step["step_id"],
                text=step["text"],
                step_type=step.get(
                    "step_type",
                    "unknown"
                )
            )

    def connect_sequential_steps(self):
        """
        Connect consecutive reasoning steps.

        For example:

            S1 → S2 → S3 → S4

        The initial relationship is "follows".

        The default weight of 1.0 is a structural placeholder,
        not a final Hodge flow value.
        """

        step_ids = list(self.graph.nodes)

        step_ids.sort()

        for source, target in zip(
            step_ids,
            step_ids[1:]
        ):
            self.add_relationship(
                source=source,
                target=target,
                relation="follows",
                weight=1.0
            )

    def build_from_extracted_result(self, extracted_result):
        """
        Build the graph from ReasoningExtractor output.

        Parameters
        ----------
        extracted_result : dict
            Output produced by ReasoningExtractor.extract()
            or ReasoningExtractor.extract_example().

        Returns
        -------
        ReasoningGraph
            The current graph object.
        """

        if "reasoning_steps" not in extracted_result:
            raise KeyError(
                "Extracted result is missing 'reasoning_steps'."
            )

        self.add_extracted_steps(
            extracted_result["reasoning_steps"]
        )

        # Connect steps according to their original order.
        self.connect_sequential_steps()

        return self

    def validate_graph(self):
        """
        Validate the structural integrity of the reasoning graph.

        Returns
        -------
        bool
            True if the graph is valid.
        """

        if self.graph.number_of_nodes() == 0:
            print(
                "Validation failed: "
                "graph contains no reasoning steps."
            )
            return False

        # Validate node attributes.
        for node, data in self.graph.nodes(data=True):

            if "text" not in data:
                print(
                    f"Validation failed: "
                    f"node {node} has no text."
                )
                return False

            if not data["text"].strip():
                print(
                    f"Validation failed: "
                    f"node {node} has empty text."
                )
                return False

            if "step_type" not in data:
                print(
                    f"Validation failed: "
                    f"node {node} has no step type."
                )
                return False

        # Validate edge attributes.
        for source, target, data in self.graph.edges(
            data=True
        ):

            if source not in self.graph.nodes:
                print(
                    f"Validation failed: "
                    f"source node {source} does not exist."
                )
                return False

            if target not in self.graph.nodes:
                print(
                    f"Validation failed: "
                    f"target node {target} does not exist."
                )
                return False

            if "relation" not in data:
                print(
                    f"Validation failed: "
                    f"edge {source}->{target} "
                    "has no relation."
                )
                return False

            if "weight" not in data:
                print(
                    f"Validation failed: "
                    f"edge {source}->{target} "
                    "has no weight."
                )
                return False

            if not isinstance(
                data["weight"],
                (int, float)
            ):
                print(
                    f"Validation failed: "
                    f"edge {source}->{target} "
                    "has non-numeric weight."
                )
                return False

        print("Graph validation successful.")

        return True

    def get_graph(self):
        """
        Return the underlying NetworkX graph.
        """

        return self.graph

    def get_reasoning_steps(self):
        """
        Return reasoning steps in graph order.

        Returns
        -------
        list[dict]
            Structured reasoning steps.
        """

        steps = []

        for node, data in self.graph.nodes(
            data=True
        ):
            steps.append(
                {
                    "step_id": node,
                    "text": data["text"],
                    "step_type": data["step_type"]
                }
            )

        return steps

    def display(self):
        """
        Display the reasoning graph.

        The visualization is intended for inspection and
        debugging of the graph representation.
        """

        if self.graph.number_of_nodes() == 0:
            print(
                "Cannot display an empty reasoning graph."
            )
            return

        plt.figure(
            figsize=(14, 8)
        )

        # Use a deterministic layout.
        positions = nx.spring_layout(
            self.graph,
            seed=42,
            k=2.0
        )

        # Wrap long reasoning text so labels remain readable.
        node_labels = {}

        for node, data in self.graph.nodes(
            data=True
        ):

            wrapped_text = textwrap.fill(
                data["text"],
                width=35
            )

            node_labels[node] = (
                f"S{node}\n"
                f"[{data['step_type']}]\n"
                f"{wrapped_text}"
            )

        # Draw graph.
        nx.draw(
            self.graph,
            positions,
            labels=node_labels,
            with_labels=True,
            node_size=7000,
            font_size=8,
            arrows=True,
            arrowsize=20
        )

        # Edge labels.
        edge_labels = {}

        for source, target, data in self.graph.edges(
            data=True
        ):

            edge_labels[
                (source, target)
            ] = (
                f"{data['relation']}\n"
                f"weight={data['weight']:.2f}"
            )

        nx.draw_networkx_edge_labels(
            self.graph,
            positions,
            edge_labels=edge_labels,
            font_size=7
        )

        plt.title(
            "Hodge-CoT Reasoning Graph",
            fontsize=13
        )

        plt.axis("off")

        plt.show()


def create_graph_from_extracted_result(
    extracted_result,
    ground_truth=None
):
    """
    Convenience function for constructing a ReasoningGraph
    directly from ReasoningExtractor output.

    Parameters
    ----------
    extracted_result : dict
        Structured reasoning result.

    ground_truth : str, optional
        Correct answer.

    Returns
    -------
    ReasoningGraph
        Constructed reasoning graph.
    """

    graph = ReasoningGraph(
        question=extracted_result.get(
            "question",
            ""
        ),
        final_answer=extracted_result[
            "final_answer"
        ],
        ground_truth=ground_truth
    )

    graph.build_from_extracted_result(
        extracted_result
    )

    return graph


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Standalone test
    # ---------------------------------------------------------

    sample_result = {
        "question": (
            "Alice has 10 apples and gives "
            "3 apples to Bob. How many apples "
            "does Alice have remaining?"
        ),

        "reasoning_steps": [
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
                    "is 10 - 3."
                )
            },
            {
                "step_id": 4,
                "text": (
                    "Therefore, Alice has "
                    "7 apples remaining."
                )
            }
        ],

        "final_answer": "7"
    }

    reasoning_graph = create_graph_from_extracted_result(
        sample_result,
        ground_truth="7"
    )

    print("\nQuestion:")
    print(reasoning_graph.question)

    print("\nFinal Answer:")
    print(reasoning_graph.final_answer)

    print("\nGround Truth:")
    print(reasoning_graph.ground_truth)

    print("\nReasoning Steps:")

    for step in reasoning_graph.get_reasoning_steps():
        print(
            f"S{step['step_id']} "
            f"[{step['step_type']}]: "
            f"{step['text']}"
        )

    print("\nValidating graph...")

    reasoning_graph.validate_graph()

    print("\nEdges:")

    for source, target, data in (
        reasoning_graph.graph.edges(data=True)
    ):
        print(
            f"S{source} -> S{target} | "
            f"relation={data['relation']} | "
            f"weight={data['weight']}"
        )

    reasoning_graph.display()