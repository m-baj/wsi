from dataclasses import dataclass
from bayes_network import BayesNetwork, Node
from tqdm import tqdm
import random


@dataclass
class MCMC:
    network: BayesNetwork
    probabilities: dict

    def inference(self, evidence, query, iterations):
        self._set_inital_nodes_values(evidence)

        counters = {
            node: {True: 0, False: 0}
            for node in self.network.nodes
            if node.name in query
        }

        for _ in range(iterations):
            random_node = self._get_random_not_evidence_node(evidence)
            random_node.value = self._sample_new_value(random_node)
            if random_node.name in query:
                counters[random_node][random_node.value] += 1

        return self._normalize(counters, iterations)

    def _set_inital_nodes_values(self, evidence):
        for node in self.network.nodes:
            if node.name in evidence:
                node.value = evidence[node.name]
            else:
                node.value = random.choice([True, False])

    def _get_random_not_evidence_node(self, evidence):
        possible_nodes = [
            node for node in self.network.nodes if node.name not in evidence.keys()
        ]
        return random.choice(possible_nodes)

    def _calculate_alpha(self, node):
        node_true = self._calc_prob_given_parents(node)
        node_false = 1 - node_true
        return 1 / (node_true + node_false)

    def _sample_new_value(self, node):
        alpha = self._calculate_alpha(node)
        prob_true = (
            alpha
            * self._calc_prob_given_parents(node)
            * self._calc_children_prob_product(node)
        )
        prob_false = (
            alpha
            * (1 - self._calc_prob_given_parents(node))
            * self._calc_children_prob_product(node)
        )
        return random.choices([True, False], weights=[prob_true, prob_false])[0]

    def _calc_prob_given_parents(self, node):
        if not node.parents:
            return self.probabilities[node.name]
        else:
            parents_values = tuple(parent.value for parent in node.parents)
            return self.probabilities[node.name][parents_values]

    def _calc_children_prob_product(self, node):
        prob_product = 1
        for child in node.children:
            prob_product *= self._calc_prob_given_parents(child)
        return prob_product

    def _normalize(self, counters, iterations):
        for node in counters.keys():
            counters[node][True] /= iterations
            counters[node][False] /= iterations
        return counters
