# persistence/gnn_layer.py
# Simple GNN layer for BrQin v5.3 – belief graph with message passing
# In-house Python – MIT License – QILLQuantum/Credo

import json
import os
import numpy as np
import networkx as nx
from typing import Dict, List, Any

class GNNLayer:
    def __init__(self, persistence_dir="brqin_persistence", graph_file="belief_graph.json"):
        self.persistence_dir = persistence_dir
        self.graph_file = os.path.join(persistence_dir, graph_file)
        os.makedirs(persistence_dir, exist_ok=True)

        # Load or initialize graph
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        else:
            self.graph = nx.DiGraph()

    def add_reflection_node(self, reflection_id: str, features: Dict[str, Any]):
        """Add node with features (oracle + memory + curvature metrics)"""
        # Flatten features to vector (simple for GNN)
        feature_vector = np.array([
            features.get('certified_energy', 0.0),
            features.get('uncertainty', 0.0),
            features.get('logical_advantage', 0.0),
            features.get('final_avg_bond', 0.0),
            features.get('growth_rate', 0.0),
            features.get('entropy_change', 0.0),
            features.get('retention_rate', 1.0),
            features.get('curvature_proxy', 0.0),
            features.get('boundary_entropy', 0.0),
            features.get('bulk_mutual_info', 0.0)
        ], dtype=float)

        self.graph.add_node(reflection_id, features=feature_vector.tolist())

        # Connect to previous reflection (chronological edge)
        if len(self.graph.nodes) > 1:
            prev_id = sorted(self.graph.nodes)[-2]  # last before this one
            influence = np.random.uniform(0.6, 0.95)  # simulated influence weight
            self.graph.add_edge(prev_id, reflection_id, weight=influence)

        # Save graph
        with open(self.graph_file, 'w') as f:
            json.dump(nx.node_link_data(self.graph), f, indent=2)

    def simple_message_passing(self, node_id: str) -> List[float]:
        """1-layer message passing: aggregate neighbor features"""
        if node_id not in self.graph:
            return np.zeros(10).tolist()

        neighbors = list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))
        if not neighbors:
            return self.graph.nodes[node_id]['features']

        # Average neighbor features + self
        agg = np.mean([self.graph.nodes[n]['features'] for n in neighbors] + [self.graph.nodes[node_id]['features']], axis=0)
        return agg.tolist()

    def get_graph_stats(self) -> Dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": np.mean([d for n, d in self.graph.degree()]),
            "density": nx.density(self.graph)
        }