from dataclasses import dataclass, field
import random


@dataclass
class Node:
    name: str
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    value: bool = None

    def __post_init__(self):
        for parent in self.parents:
            parent.children.append(self)

        for child in self.children:
            child.parents.append(self)

    def __hash__(self):
        return hash(self.name)


@dataclass
class BayesNetwork:
    nodes: list

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, parent, child):
        parent.children.append(child)
        child.parents.append(parent)
