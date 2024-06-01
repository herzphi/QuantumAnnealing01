import numpy as np
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave_networkx import traveling_salesman, traveling_salesman_qubo

# Define the distances between cities
cities = ["A", "B", "C", "D"]
distances = {
    ("A", "B"): 10,
    ("A", "C"): 15,
    ("A", "D"): 20,
    ("B", "C"): 35,
    ("B", "D"): 25,
    ("C", "D"): 30,
}

# Create a complete graph with the cities and their distances
G = nx.complete_graph(len(cities))
nx.set_edge_attributes(
    G,
    {
        (i, j): distances[(cities[i], cities[j])]
        for i in range(len(cities))
        for j in range(i + 1, len(cities))
    },
    "weight",
)

# Convert the TSP to a QUBO
qubo = traveling_salesman_qubo(G, weight="weight")

# Use D-Wave's quantum annealer to solve the QUBO
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(qubo, num_reads=100)

# Extract the best solution
sample = response.first.sample
route = [cities[i] for i in sample if sample[i] == 1]

# Print the best route and its distance
total_distance = sum(distances[(route[i], route[i + 1])] for i in range(len(route) - 1))
print(f"Best route: {' -> '.join(route)}")
print(f"Total distance: {total_distance}")
