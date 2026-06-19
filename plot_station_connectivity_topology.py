import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
ADJ_CSV = os.path.join(ROOT, "AD_matrix_trimmed_common.csv")
OUT_DIR = os.path.join(ROOT, "analysis_results")
OUT_PATH = os.path.join(OUT_DIR, "metro_station_connectivity_topology.png")
OUT_LABELED_PATH = os.path.join(OUT_DIR, "metro_station_connectivity_topology_labeled.png")


def build_graph():
    adj_df = pd.read_csv(ADJ_CSV, index_col=0)
    names = list(adj_df.index)
    adj = adj_df.values

    graph = nx.DiGraph()
    for idx, name in enumerate(names):
        graph.add_node(idx, label=name)

    rows, cols = (adj > 0).nonzero()
    for src, dst in zip(rows, cols):
        graph.add_edge(int(src), int(dst))

    return graph


def draw_graph(graph, out_path, labeled=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    undirected = graph.to_undirected()
    pos = nx.spring_layout(undirected, seed=42, k=0.22, iterations=300)

    degree = dict(undirected.degree())
    node_sizes = [10 + degree[n] * 14 for n in undirected.nodes()]

    fig, ax = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_edges(
        undirected,
        pos,
        ax=ax,
        width=0.35,
        alpha=0.14,
        edge_color="#4b5563",
    )
    nodes = nx.draw_networkx_nodes(
        undirected,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=list(degree.values()),
        cmap="viridis",
        linewidths=0.15,
        edgecolors="white",
    )

    if labeled:
        top_nodes = sorted(degree, key=degree.get, reverse=True)[:20]
        labels = {n: graph.nodes[n]["label"] for n in top_nodes}
        nx.draw_networkx_labels(
            undirected,
            pos,
            labels=labels,
            font_size=7,
            font_color="black",
            ax=ax,
        )

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Undirected degree")

    ax.set_title("Metro Station Connectivity Topology")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    graph = build_graph()
    draw_graph(graph, OUT_PATH, labeled=False)
    draw_graph(graph, OUT_LABELED_PATH, labeled=True)
    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {OUT_LABELED_PATH}")


if __name__ == "__main__":
    main()
