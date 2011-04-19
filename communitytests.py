import unittest
import networkx as nx
import community as co
import random

    
class ModularityTest(unittest.TestCase):
    numtest = 100
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_allin_is_zero(self):
        """it test that everyone in one community has a modularity of 0"""
        for i in range(self.numtest) :
            g = nx.erdos_renyi_graph(50, 0.1)
            part = dict([])
            for node in g :
                part[node] = 0
            self.assertEqual(co.modularity(part, g), 0)
            
    def test_range(self) :
        """test that modularity is always between -1 and 1"""
        for i in range(self.numtest) :
            g = nx.erdos_renyi_graph(50, 0.1)
            part = dict([])
            for node in g :
                part[node] = random.randint(0, self.numtest/10)
            mod = co.modularity(part, g)
            self.assertGreaterEqual(mod, -1)
            self.assertLessEqual(mod, 1)
            
    def test_bad_graph_input(self) :
        """modularity is only defined with undirected graph"""
        g = nx.erdos_renyi_graph(50, 0.1, directed=True)
        part = dict([])
        for node in g :
            part[node] = 0
        self.assertRaises(TypeError, co.modularity, part, g)

    def test_empty_graph_input(self) :
        """modularity of a graph without links is undefined"""
        g = nx.Graph()
        g.add_nodes_from(range(10))
        part = dict([])
        for node in g :
            part[node] = 0
        self.assertRaises(ValueError, co.modularity, part, g)
        
    def test_bad_partition_input(self) :
        """modularity is undefined when some nodes are not in a community"""
        g = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for count, node in enumerate(g) :
            part[node] = 0
            if count == 40 :
                break    
        self.assertRaises(KeyError, co.modularity, part, g)


    #These are known values taken from the paper
    #1. Bartheemy, M. & Fortunato, S. Resolution limit in community detection. Proceedings of the National Academy of Sciences of the United States of America 104, 36-41(2007).
    def test_disjoint_clique(self) :
        """"
        A group of num_clique of size size_clique disjoint, should maximize the modularity
        and have a modularity of 1 - 1/ num_clique
        """
        for num_test in range(self.numtest) :
            size_clique = random.randint(5, 20)
            num_clique = random.randint(5, 20)
            g = nx.Graph()
            for i in range(num_clique) :
                clique_i = nx.complete_graph(size_clique)
                g = nx.union(g, clique_i, rename=("",str(i)+"_"))
            part = dict([])
            for node in g :
                part[node] = node.split("_")[0].strip()
            mod = co.modularity(part, g)
            self.assertAlmostEqual(mod, 1. - 1./float(num_clique),  msg = "Num clique: " + str(num_clique) + " size_clique: " + str(size_clique))

            
    def test_ring_clique(self) :
        """"
        then, a group of num_clique of size size_clique connected with only two links to other in a ring
        have a modularity of 1 - 1/ num_clique - num_clique / num_links
        """
        for num_test in range(self.numtest) :
            size_clique = random.randint(5, 20)
            num_clique = random.randint(5, 20)
            g = nx.Graph()
            for i in range(num_clique) :
                clique_i = nx.complete_graph(size_clique)
                g = nx.union(g, clique_i, rename=("",str(i)+"_"))
                if i > 0 :
                    g.add_edge(str(i)+"_0", str(i-1)+"_1")
            g.add_edge("0_0", str(num_clique-1)+"_1")
            part = dict([])
            for node in g :
                part[node] = node.split("_")[0].strip()
            mod = co.modularity(part, g)
            self.assertAlmostEqual(mod, 1. - 1./float(num_clique) - float(num_clique) / float(g.number_of_edges()), msg = "Num clique: " + str(num_clique) + " size_clique: " + str(size_clique) )

        


class BestPartitionTest(unittest.TestCase):
    numtest = 100

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bad_graph_input(self) :
        """best_partition is only defined with undirected graph"""
        g = nx.erdos_renyi_graph(50, 0.1, directed=True)
        self.assertRaises(TypeError, co.best_partition,  g)

    def test_empty_graph_input(self) :
        """best_partition of a graph without links is undefined"""
        g = nx.Graph()
        g.add_nodes_from(range(10))
        self.assertRaises(ValueError, co.best_partition,  g)

            
    

if __name__ == '__main__':
    unittest.main()