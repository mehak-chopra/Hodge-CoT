import networkx as nx
import matplotlib.pyplot as plt


def create_reasoning_graph():
    """
    Create a simple reasoning graph from manually defined
    reasoning steps.
    """

    # Example reasoning chain
    reasoning_steps = {
        1: "Alice initially has 10 apples.",
        2: "Alice gives 3 apples to Bob.",
        3: "The number of apples remaining is 10 - 3.",
        4: "Therefore, Alice has 7 apples remaining."
    }

    # Create a directed graph
    graph = nx.DiGraph()

    # Add reasoning steps as nodes
    for step_id, reasoning in reasoning_steps.items():
        graph.add_node(
            step_id,
            reasoning=reasoning
        )

    # Add relationships between reasoning steps
    graph.add_edge(1, 2, relation="supports")
    graph.add_edge(2, 3, relation="supports")
    graph.add_edge(3, 4, relation="supports")

    return graph


def display_reasoning_graph(graph):
    """
    Display the reasoning graph visually.
    """

    # Position nodes according to the reasoning flow
    positions = nx.spring_layout(graph, seed=42)

    # Create labels containing step number and reasoning
    node_labels = {
        node: f"S{node}\n{graph.nodes[node]['reasoning']}"
        for node in graph.nodes
    }

    # Draw the graph
    nx.draw(
        graph,
        positions,
        with_labels=True,
        labels=node_labels,
        node_size=5000,
        font_size=8,
        arrows=True
    )

    # Display relationship labels
    edge_labels = nx.get_edge_attributes(graph, "relation")

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title("Hodge-CoT Reasoning Graph")
    plt.show()


if __name__ == "__main__":
    reasoning_graph = create_reasoning_graph()
    display_reasoning_graph(reasoning_graph)