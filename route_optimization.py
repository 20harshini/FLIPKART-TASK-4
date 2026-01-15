import networkx as nx

def optimize_route():
    G = nx.Graph()

    G.add_edge("Warehouse", "A", weight=10)
    G.add_edge("A", "B", weight=5)
    G.add_edge("B", "Customer", weight=8)
    G.add_edge("Warehouse", "C", weight=15)
    G.add_edge("C", "Customer", weight=6)

    path = nx.shortest_path(G, source="Warehouse", target="Customer", weight="weight")
    distance = nx.shortest_path_length(G, source="Warehouse", target="Customer", weight="weight")

    return path, distance

if __name__ == "__main__":
    print(optimize_route())
