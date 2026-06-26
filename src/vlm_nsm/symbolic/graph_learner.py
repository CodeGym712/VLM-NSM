# GraphLearner — learns state transition graph from demonstration traces.
# Adapted from NSM project for VLM-NSM.
# Removed matplotlib, added JSON export, adapted for predicate-based states.

import json
import networkx as nx


class GraphLearner:
    """Learns a directed graph from (state, action_label, next_state) triples.

    Each state is a frozenset of predicate tuples, e.g.:
        frozenset({("on","cube1","cube2"), ("clear","cube1"), ("handempty",)})

    Each action_label is a string, e.g. "pick_place".

    Usage::

        learner = GraphLearner("generated/results/symbolic/stacking")
        for episode in demos:
            learner.learn(state_before, "pick_place", state_after)
        learner.save()
        # → stacking_graph.dfa, stacking_bisimulation.dfa, stacking_graph.json
    """

    def __init__(self, output_prefix: str):
        self.graph = nx.DiGraph()
        self.bisimulation = None
        self.output_prefix = output_prefix
        self.state_mapping: dict[str, frozenset] = {}
        self.state_set: set[frozenset] = set()

    # ── State management ─────────────────────────────────────────────────

    def _to_frozenset(self, state) -> frozenset:
        """Convert a state dict or set to a hashable frozenset of tuples."""
        if isinstance(state, frozenset):
            return state
        if isinstance(state, dict):
            items = []
            for k, v in state.items():
                if v:  # only include True predicates
                    # Parse "on(cube1,cube2)" → ("on","cube1","cube2")
                    name = k.split("(")[0] if "(" in k else k
                    args = tuple(k.split("(")[1].rstrip(")").split(",")) if "(" in k else ()
                    items.append((name,) + args)
            return frozenset(items)
        if isinstance(state, (set, list)):
            items = []
            for item in state:
                if isinstance(item, (tuple, list)):
                    items.append(tuple(str(x) for x in item))
                elif isinstance(item, str):
                    name = item.split("(")[0] if "(" in item else item
                    args = tuple(item.split("(")[1].rstrip(")").split(",")) if "(" in item else ()
                    items.append((name,) + args)
            return frozenset(items)
        return frozenset()

    def _add_state(self, state) -> str:
        """Add state to graph, return node name (str index)."""
        fs = self._to_frozenset(state)
        if fs not in self.state_set:
            name = str(len(self.state_mapping))
            self.state_mapping[name] = fs
            self.state_set.add(fs)
            self.graph.add_node(name)
        else:
            name = next(k for k, v in self.state_mapping.items() if v == fs)
        return name

    # ── Learning ─────────────────────────────────────────────────────────

    def learn(self, state_before, action_label: str, state_after):
        """Add one transition to the graph."""
        s = self._add_state(state_before)
        ns = self._add_state(state_after)
        if s is None or ns is None:
            return
        self.graph.add_edge(s, ns, label=action_label)

    def learn_from_triples(self, triples: list[tuple]):
        """Bulk learn from a list of (state_before, action, state_after) triples."""
        for state_before, action, state_after in triples:
            self.learn(state_before, action, state_after)
        self.bisimulation = self._compute_bisimulation()

    # ── Bisimulation ─────────────────────────────────────────────────────

    def _compute_bisimulation(self):
        """Compute minimum bisimulation of the graph."""
        G = self.graph
        partition = self._bisect(G)

        B = nx.DiGraph()
        for i, block in enumerate(partition):
            B.add_node(i)

        for i, block1 in enumerate(partition):
            for j, block2 in enumerate(partition):
                for u in block1:
                    for v in block2:
                        if G.has_edge(u, v):
                            B.add_edge(i, j, label=G.edges[u, v]['label'])
                            break
        return B

    def _bisect(self, G):
        """Bisect algorithm — partition states by outgoing edge labels."""
        partition = [{node} for node in G.nodes()]
        worklist = partition.copy()

        while worklist:
            B = worklist.pop()
            edge_labels = set(
                data['label'] for node in B for _, data in G[node].items())
            for label in edge_labels:
                B1 = {node for node in B if any(
                    data['label'] == label for _, data in G[node].items())}
                B2 = B - B1
                if B1 and B2:
                    partition.remove(B)
                    partition.extend([B1, B2])
                    worklist.extend([B1, B2])
        return partition

    # ── DFA Export ───────────────────────────────────────────────────────

    def _write_dfa(self, filepath: str, G: nx.DiGraph | None = None):
        """Write graph in DFA format."""
        if G is None:
            G = self.bisimulation or self.graph
        with open(filepath, 'w') as f:
            f.write(f"dfa {G.number_of_nodes()} -1\n")
            labels = sorted(set(nx.get_edge_attributes(G, 'label').values()))
            f.write(f"{len(labels)} {' '.join(labels)}\n")
            f.write("1 0\n")
            for node in G.nodes():
                edges = list(G.out_edges(node, data=True))
                f.write(f"{len(edges)}")
                for _, v, data in edges:
                    f.write(f" {data.get('label', 'MOVE')} {v}")
                f.write("\n")

    # ── Save ─────────────────────────────────────────────────────────────

    def save(self):
        """Save graph.dfa, bisimulation.dfa, and graph.json."""
        prefix = self.output_prefix
        self._write_dfa(f"{prefix}_graph.dfa", self.graph)
        if self.bisimulation:
            self._write_dfa(f"{prefix}_bisimulation.dfa", self.bisimulation)

        # JSON summary
        summary = {
            "num_states": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_bisim_states": self.bisimulation.number_of_nodes() if self.bisimulation else 0,
            "states": {
                name: sorted(str(p) for p in predicates)
                for name, predicates in self.state_mapping.items()
            },
            "edges": [
                {"from": u, "to": v, "label": d["label"]}
                for u, v, d in self.graph.edges(data=True)
            ],
            "bisim_edges": [
                {"from": u, "to": v, "label": d["label"]}
                for u, v, d in self.bisimulation.edges(data=True)
            ] if self.bisimulation else [],
        }
        with open(f"{prefix}_graph.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges → {prefix}_graph.dfa")
        if self.bisimulation:
            print(f"Bisimulation: {self.bisimulation.number_of_nodes()} nodes → "
                  f"{prefix}_bisimulation.dfa")
        print(f"JSON → {prefix}_graph.json")
