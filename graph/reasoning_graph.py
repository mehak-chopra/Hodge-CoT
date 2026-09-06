import networkx as nx
import matplotlib.pyplot as plt


class ReasoningGraph:
    """
    Represents a language model reasoning trace as a directed graph.

    Each reasoning step is represented as a node.
    Relationships between reasoning steps are represented as
    directed, weighted edges.
    """

    def __init__(self, question, final_answer, ground_truth=None):
        """
        Initialize a reasoning graph.

        Parameters
        ----------
        question : str
            The original reasoning question.

        final_answer : str
            The answer produced by the model.

        ground_truth : str, optional
            The correct answer from the dataset, if available.
        """

        self.question = question
        self.final_answer = final_answer
        self.ground_truth = ground_truth

        # NetworkX directed graph
        self.graph = nx.DiGraph()

    def add_reasoning_step(self, step_id, text, step_type="inference"):
        """
        Add a reasoning step as a node.

        Parameters
        ----------
        step_id : int
            Unique identifier for the reasoning step.

        text : str
            The actual reasoning statement.

        step_type : str
            Type of reasoning step, such as:
            premise, operation, inference, conclusion.
        """

        self.graph.add_node(
            step_id,
            text=text,
            step_type=step_type
        )

    def add_relationship(
        self,
        source,
        target,
        relation="supports",
        weight=1.0
    ):
        """
        Add a directed relationship between two reasoning steps.

        Parameters
        ----------
        source : int
            ID of the source reasoning step.

        target : int
            ID of the target reasoning step.

        relation : str
            Type of relationship, for example:
            supports, contradicts, depends_on, implies.

        weight : float
            Numerical strength of the relationship.
        """

        self.graph.add_edge(
            source,
            target,
            relation=relation,
            weight=weight
        )

    def validate_graph(self):
        """
        Validate the reasoning graph structure.

        Returns
        -------
        bool
            True if the graph is valid, otherwise False.
        """

        if self.graph.number_of_nodes() == 0:
            print("Validation failed: graph contains no reasoning steps.")
            return False

        # Check that every edge connects existing nodes.
        for source, target in self.graph.edges():

            if source not in self.graph.nodes:
                print(f"Validation failed: source node {source} does not exist.")
                return False

            if target not in self.graph.nodes:
                print(f"Validation failed: target node {target} does not exist.")
                return False

        # Check that every edge has required attributes.
        for source, target, data in self.graph.edges(data=True):

            if "relation" not in data:
                print(
                    f"Validation failed: edge {source}->{target} "
                    "has no relation."
                )
                return False

            if "weight" not in data:
                print(
                    f"Validation failed: edge {source}->{target} "
                    "has no weight."
                )
                return False

        print("Graph validation successful.")
        return True

    def get_graph(self):
        """
        Return the underlying NetworkX graph.
        """

        return self.graph

    def display(self):
        """
        Display the reasoning graph visually.
        """

        if self.graph.number_of_nodes() == 0:
            print("Cannot display an empty reasoning graph.")
            return

        # Create a reproducible layout.
        positions = nx.spring_layout(
            self.graph,
            seed=42
        )

        # Create node labels.
        node_labels = {}

        for node, data in self.graph.nodes(data=True):

            step_type = data.get("step_type", "unknown")
            text = data.get("text", "")

            node_labels[node] = (
                f"S{node}\n"
                f"[{step_type}]\n"
                f"{text}"
            )

        # Draw nodes and directed edges.
        nx.draw(
            self.graph,
            positions,
            with_labels=True,
            labels=node_labels,
            node_size=6000,
            font_size=8,
            arrows=True,
            arrowsize=20
        )

        # Create edge labels.
        edge_labels = {}

        for source, target, data in self.graph.edges(data=True):

            relation = data.get("relation", "unknown")
            weight = data.get("weight", 1.0)

            edge_labels[(source, target)] = (
                f"{relation}\n"
                f"weight={weight:.2f}"
            )

        # Draw relationship labels.
        nx.draw_networkx_edge_labels(
            self.graph,
            positions,
            edge_labels=edge_labels,
            font_size=7
        )

        # Display graph information.
        plt.title(
            "Hodge-CoT Reasoning Graph",
            fontsize=12
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()


def create_reasoning_graph():
    """
    Create a sample reasoning graph.

    This manually constructed example is used to test the
    graph representation before connecting it to a dataset
    and language model.
    """

    # ---------------------------------------------------------
    # 1. Problem information
    # ---------------------------------------------------------

    question = (
        "Alice has 10 apples and gives 3 apples to Bob. "
        "How many apples does Alice have remaining?"
    )

    final_answer = "7"
    ground_truth = "7"

    # Create the reasoning graph object.
    reasoning_graph = ReasoningGraph(
        question=question,
        final_answer=final_answer,
        ground_truth=ground_truth
    )

    # ---------------------------------------------------------
    # 2. Add reasoning steps
    # ---------------------------------------------------------

    reasoning_graph.add_reasoning_step(
        step_id=1,
        text="Alice initially has 10 apples.",
        step_type="premise"
    )

    reasoning_graph.add_reasoning_step(
        step_id=2,
        text="Alice gives 3 apples to Bob.",
        step_type="operation"
    )

    reasoning_graph.add_reasoning_step(
        step_id=3,
        text="The number of apples remaining is 10 - 3.",
        step_type="inference"
    )

    reasoning_graph.add_reasoning_step(
        step_id=4,
        text="Therefore, Alice has 7 apples remaining.",
        step_type="conclusion"
    )

    # ---------------------------------------------------------
    # 3. Add relationships between reasoning steps
    # ---------------------------------------------------------

    reasoning_graph.add_relationship(
        source=1,
        target=2,
        relation="supports",
        weight=1.0
    )

    reasoning_graph.add_relationship(
        source=2,
        target=3,
        relation="supports",
        weight=1.0
    )

    reasoning_graph.add_relationship(
        source=3,
        target=4,
        relation="implies",
        weight=1.0
    )

    return reasoning_graph


if __name__ == "__main__":

    # Create the sample reasoning graph.
    reasoning_graph = create_reasoning_graph()

    # Print basic problem information.
    print("\nQuestion:")
    print(reasoning_graph.question)

    print("\nFinal Answer:")
    print(reasoning_graph.final_answer)

    print("\nGround Truth:")
    print(reasoning_graph.ground_truth)

    # Validate the graph.
    print("\nValidating graph...")
    reasoning_graph.validate_graph()

    # Display the graph.
    reasoning_graph.display()