# coding=utf-8
"""
Test for community package
"""
import unittest
import random

import networkx as nx
import numpy

import community as co
from community.community_louvain import __randomize as randomize, __renumber as renumber


def girvan_graphs(zout):
    """
    Create a graph of 128 vertices, 4 communities, like in
    Community Structure in  social and biological networks.
    Girvan newman, 2002. PNAS June, vol 99 n 12

    community is node_1 modulo 4
    """

    pout = float(zout) / 96.
    pin = (16. - pout * 96.) / 31.
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    for node_1 in graph.nodes():
        for node_2 in graph.nodes():
            if node_1 < node_2:
                val = random.random()
                if node_1 % 4 == node_2 % 4:
                    # nodes belong to the same community
                    if val < pin:
                        graph.add_edge(node_1, node_2)

                else:
                    if val < pout:
                        graph.add_edge(node_1, node_2)
    return graph


class ModularityTest(unittest.TestCase):
    """
    Tests for modularity function
    """
    number_of_tests = 10

    def test_allin_is_zero(self):
        """it test that everyone in one community has a modularity of 0"""
        for _ in range(self.number_of_tests):
            graph = nx.erdos_renyi_graph(50, 0.1)
            part = dict([])
            for node in graph:
                part[node] = 0
            self.assertEqual(co.modularity(part, graph), 0)

    def test_range(self):
        """test that modularity is always between -1 and 1"""
        for _ in range(self.number_of_tests):
            graph = nx.erdos_renyi_graph(50, 0.1)
            part = dict([])
            for node in graph:
                part[node] = random.randint(0, self.number_of_tests / 10)
            mod = co.modularity(part, graph)
            self.assertGreaterEqual(mod, -1)
            self.assertLessEqual(mod, 1)

    def test_bad_graph_input(self):
        """modularity is only defined with undirected graph"""
        graph = nx.erdos_renyi_graph(50, 0.1, directed=True)
        part = dict([])
        for node in graph:
            part[node] = 0
        self.assertRaises(TypeError, co.modularity, part, graph)

    def test_empty_graph_input(self):
        """modularity of a graph without links is undefined"""
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        part = dict([])
        for node in graph:
            part[node] = 0
        self.assertRaises(ValueError, co.modularity, part, graph)

    def test_bad_partition_input(self):
        """modularity is undefined when some nodes are not in a community"""
        graph = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for count, node in enumerate(graph):
            part[node] = 0
            if count == 40:
                break
        self.assertRaises(KeyError, co.modularity, part, graph)

    # These are known values taken from the paper
    # 1. Bartheemy, M. & Fortunato, S. Resolution limit in community detection.
    # Proceedings of the National Academy of Sciences of the United States of
    # America 104, 36-41(2007).
    def test_disjoint_clique(self):
        """"
        A group of num_clique of size size_clique disjoint, should maximize
        the modularity and have a modularity of 1 - 1/ num_clique
        """
        for _ in range(self.number_of_tests):
            size_clique = random.randint(5, 20)
            num_clique = random.randint(5, 20)
            graph = nx.Graph()
            for i in range(num_clique):
                clique_i = nx.complete_graph(size_clique)
                graph = nx.union(graph, clique_i, rename=("", str(i) + "_"))
            part = dict([])
            for node in graph:
                part[node] = node.split("_")[0].strip()
            mod = co.modularity(part, graph)
            self.assertAlmostEqual(mod, 1. - 1. / float(num_clique))

    def test_ring_clique(self):
        """"
        then, a group of num_clique of size size_clique connected with only
        two links to other in a ring have a modularity of
        1 - 1/ num_clique - num_clique / num_links
        """
        for _ in range(self.number_of_tests):
            size_clique = random.randint(5, 20)
            num_clique = random.randint(5, 20)
            graph = nx.Graph()
            for i in range(num_clique):
                clique_i = nx.complete_graph(size_clique)
                graph = nx.union(graph, clique_i, rename=("", str(i) + "_"))
                if i > 0:
                    graph.add_edge(str(i) + "_0", str(i - 1) + "_1")
            graph.add_edge("0_0", str(num_clique - 1) + "_1")
            part = dict([])
            for node in graph:
                part[node] = node.split("_")[0].strip()
            mod = co.modularity(part, graph)
            self.assertAlmostEqual(mod, 1. - 1. / float(num_clique) - float(
                num_clique) / float(graph.number_of_edges()))


class BestPartitionTest(unittest.TestCase):
    """
    Test for best_partition function
    """
    number_of_tests = 10

    def test_bad_graph_input(self):
        """best_partition is only defined with undirected graph"""
        graph = nx.erdos_renyi_graph(50, 0.1, directed=True)
        self.assertRaises(TypeError, co.best_partition, graph)

    def test_girvan(self):
        """
        Test that community found are good using Girvan & Newman benchmark
        """
        graph = girvan_graphs(3)  # use small zout otherwise results may change
        part = co.best_partition(graph)
        for node, com in part.items():
            self.assertEqual(com, part[node % 4])

    def test_ring(self):
        """
        Test that community found are good using a ring of cliques
        """
        for _ in range(self.number_of_tests):
            size_clique = random.randint(5, 20)
            num_clique = random.randint(5, 20)
            graph = nx.Graph()
            for i in range(num_clique):
                clique_i = nx.complete_graph(size_clique)
                graph = nx.union(graph, clique_i, rename=("", str(i) + "_"))
                if i > 0:
                    graph.add_edge(str(i) + "_0", str(i - 1) + "_1")
            graph.add_edge("0_0", str(num_clique - 1) + "_1")
            part = co.best_partition(graph)

            for clique in range(num_clique):
                part_name = part[str(clique) + "_0"]
                for node in range(size_clique):
                    expected = part[str(clique) + "_" + str(node)]
                    self.assertEqual(part_name, expected)

    def test_all_nodes(self):
        """
        Test that all nodes are in a community
        """
        graph = nx.erdos_renyi_graph(50, 0.1)
        part = co.best_partition(graph)
        for node in graph.nodes():
            self.assertTrue(node in part)

    def test_karate(self):
        """"test modularity on Zachary's karate club"""
        graph = nx.karate_club_graph()
        part = co.best_partition(graph, random_state=0)
        self.assertTrue(co.modularity(part, graph) > 0.41)

        for e1, e2 in graph.edges():
            graph[e1][e2]["test_weight"] = 1.

        part_weight = co.best_partition(graph, weight="test_weight", random_state=0)
        self.assertAlmostEqual(co.modularity(part, graph),
                               co.modularity(part_weight,
                                             graph,
                                             "test_weight"), places=2)

        part_res_low = co.best_partition(graph, resolution=0.1)
        self.assertTrue(
            len(set(part.values())) < len(set(part_res_low.values())))


class InducedGraphTest(unittest.TestCase):
    """
    Test the induce graph
    """
    def test_nodes(self):
        """
        Test that result nodes are the communities
        """
        graph = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in graph.nodes():
            part[node] = node % 5
        self.assertSetEqual(set(part.values()),
                            set(co.induced_graph(part, graph).nodes()))

    def test_weight(self):
        """
        Test that total edge weight does not change
        """
        graph = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in graph.nodes():
            part[node] = node % 5
        self.assertEqual(graph.size(weight='weight'),
                         co.induced_graph(part, graph).size(weight='weight'))

        for e1, e2 in graph.edges():
            graph[e1][e2]["test_weight"] = 1.

        induced = co.induced_graph(part, graph, "test_weight")
        self.assertEqual(graph.size(weight='test_weight'),
                         induced.size(weight='test_weight'))

    def test_unique(self):
        """
        Test that the induced graph is the same when all nodes are alone
        """
        graph = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        ind = co.induced_graph(part, graph)
        self.assertTrue(nx.is_isomorphic(graph, ind))

    def test_clique(self):
        """
        Test that a complete graph of size 2*graph_size has the right behavior
        when split in two
        """
        graph_size = 5
        graph = nx.complete_graph(2 * graph_size)
        part = dict([])
        for node in graph.nodes():
            part[node] = node % 2
        ind = co.induced_graph(part, graph)
        goal = nx.Graph()
        edges = [(0, 1, graph_size ** 2),
                 (0, 0, graph_size * (graph_size - 1) / 2),
                 (1, 1, graph_size * (graph_size - 1) / 2)]
        goal.add_weighted_edges_from(edges)
        self.assertTrue(nx.is_isomorphic(ind, goal))


class GenerateDendrogramTest(unittest.TestCase):
    """
    Test dendogram generation
    """
    def test_bad_graph_input(self):
        """generate_dendrogram is only defined with undirected graph"""
        graph = nx.erdos_renyi_graph(50, 0.1, directed=True)
        self.assertRaises(TypeError, co.best_partition, graph)

    def test_modularity_increase(self):
        """
        Generate a dendrogram and test that modularity is always increasing
        """
        graph = nx.erdos_renyi_graph(1000, 0.01)
        dendo = co.generate_dendrogram(graph)
        mods = [co.modularity(co.partition_at_level(dendo, level), graph)
                for level in range(len(dendo))]
        self.assertListEqual(mods, sorted(mods))

    def test_nodes_stay_together(self):
        """
        Test that two nodes in the same community at one level stay in the
        same at higher level
        """
        graph = nx.erdos_renyi_graph(500, 0.01)
        dendo = co.generate_dendrogram(graph)
        parts = dict([])
        for level in range(len(dendo)):
            parts[level] = co.partition_at_level(dendo, level)
        for level in range(len(dendo) - 1):
            part_1 = parts[level]
            part_2 = parts[level + 1]
            coms = set(part_1.values())
            for com in coms:
                comhigher = [part_2[node]
                             for node, comnode in part_1.items()
                             if comnode == com]
                self.assertEqual(len(set(comhigher)), 1)


class NodeTypeTest(unittest.TestCase):
    def test_multiple_node_types(self):
        """
        Test that a graph with nodes of different Python types still gets
        correct results
        """
        node01 = 6
        node02 = 'hello'
        node11 = False
        node12 = lambda: 0
        node13 = 22.2
        edges = [(node01, node02), (node11, node12), (node12, node13)]
        graph = nx.Graph(edges)
        partition = co.best_partition(graph)
        for (a, b) in edges:
            self.assertEqual(partition[a], partition[b])
        self.assertNotEqual(partition[node01], partition[node11])


class RandomizeTest(unittest.TestCase):
    """Test the __randomize utility function"""

    def test_randomize_handles_list_items(self):
        """Test that Lists are randomized correctly"""
        l = list(range(100))
        random_state = numpy.random.RandomState()
        randomized_items = randomize(l, random_state)
        self.assertNotEqual(l, randomized_items, "List was not randomized")
        self.assertEqual(set(l), set(randomized_items),
                         "Input items and randomized items are not equal sets")
        self.assertEqual(sorted(l), l, "Input list was changed")

    def test_randomize_handles_dict_items(self):
        """Test that Dict#items() are randomized correctly"""
        d = {"1": 1, 2: 2, "3": "3"}
        random_state = numpy.random.RandomState()
        randomized_items = randomize(d.items(), random_state)
        self.assertEqual(set(d.items()), set(randomized_items),
                         "Input items and randomized items are not equal sets")

    def test_randomize_handles_mixed_items(self):
        """Test that Lists of mixed types are randomized correctly"""
        random_state = numpy.random.RandomState()
        items = [True, 1, 1.0, lambda x: 1, (2,), "test"]
        randomized_items = randomize(items, random_state)
        self.assertEqual(set(items), set(randomized_items),
                         "Input items and randomized items are not equal sets")

    def test_randomize_handles_iterators(self):
        """Test that Iterators are randomized correctly"""
        it = iter(range(10))
        random_state = numpy.random.RandomState()
        randomized_items = randomize(it, random_state)
        self.assertEqual(set(range(10)), set(randomized_items),
                         "Input items and randomized items are not equal sets")


class RenumberTest(unittest.TestCase):
    """Test the __renumber utility function"""

    def test_renumber_unchanged(self):
        """Test that a partition is not renumbered unnecessarily"""
        in_dict = {0: 1, 1: 0}
        self.assertEqual(in_dict, renumber(in_dict))

    def test_renumber_changed(self):
        """Test that a partition is changed when necessary"""
        in_dict = {0: 0, 1: 3, 2: 4, 3: 6}
        expected_dict = {0: 0, 1: 3, 2: 1, 3: 2}
        self.assertEqual(expected_dict, renumber(in_dict))


if __name__ == '__main__':
    unittest.main()
