import unittest
import networkx as nx
import community as co
import random

def girvan_graphs(zout) :
    """
    Create a graph of 128 vertices, 4 communities, like in
    Community Structure in  social and biological networks.
    Girvan newman, 2002. PNAS June, vol 99 n 12

    community is node modulo 4
    """

    pout = float(zout)/96.
    pin = (16.-pout*96.)/31.
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    for x in graph.nodes() :
        for y in graph.nodes() :
            if x < y :
                val = random.random()
                if x % 4 == y % 4 :
                    #nodes belong to the same community
                    if val < pin :
                        graph.add_edge(x, y)

                else :
                    if val < pout :
                        graph.add_edge(x, y)
    return graph
    
class ModularityTest(unittest.TestCase):
    
    numtest = 10
    
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
    numtest = 10

    def test_bad_graph_input(self) :
        """best_partition is only defined with undirected graph"""
        g = nx.erdos_renyi_graph(50, 0.1, directed=True)
        self.assertRaises(TypeError, co.best_partition,  g)

    def test_girvan(self) :
        """
        Test that community found are good using Girvan & Newman benchmark
        """
        g = girvan_graphs(4)#use small zout, with high zout results may change
        part = co.best_partition(g)
        for node, com in part.iteritems() :
            self.assertEqual(com, part[node%4])

    def test_ring(self) :
        """
        Test that community found are good using a ring of cliques
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
            part = co.best_partition(g)

            for clique in range(num_clique) :
                p = part[str(clique) + "_0"]
                for node in range(size_clique) :
                    self.assertEqual(p, part[str(clique) + "_" + str(node)])
                    
    def test_allnodes(self) :
        """
        Test that all nodes are in a community
        """
        g = nx.erdos_renyi_graph(50, 0.1)
        part = co.best_partition(g)
        for node in g.nodes() :
            self.assert_(part.has_key(node))
        
            


class InducedGraphTest(unittest.TestCase):

    def test_nodes(self) :
        """
        Test that result nodes are the communities
        """
        g = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in g.nodes() :
            part[node] = node % 5
        self.assertSetEqual(set(part.values()), set(co.induced_graph(part, g).nodes()))
        
    def test_weight(self) :
        """
        Test that total edge weight does not change
        """
        g = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in g.nodes() :
            part[node] = node % 5
        self.assertEqual(g.size(weight = 'weight'), co.induced_graph(part, g).size(weight = 'weight'))

    def test_uniq(self) :
        """
        Test that the induced graph is the same when all nodes are alone
        """
        g = nx.erdos_renyi_graph(50, 0.1)
        part = dict([])
        for node in g.nodes() :
            part[node] = node
        ind = co.induced_graph(part, g)
        self.assert_(nx.is_isomorphic(g, ind))

    def test_clique(self):
        """
        Test that a complet graph of size 2*n has the right behavior when split in two
        """
        n = 5
        g = nx.complete_graph(2*n)
        part = dict([])
        for node in g.nodes() :
            part[node] = node % 2
        ind = co.induced_graph(part, g)
        goal = nx.Graph()
        goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])
        self.assert_(nx.is_isomorphic(ind, goal))


class PartitionAtLevelTest(unittest.TestCase):
    pass
        
class GenerateDendogramTest(unittest.TestCase):
    def test_bad_graph_input(self) :
        """generate_dendogram is only defined with undirected graph"""
        g = nx.erdos_renyi_graph(50, 0.1, directed=True)
        self.assertRaises(TypeError, co.best_partition,  g)

    def test_modularity_increase(self):
        """
        Generate a dendogram and test that modularity is always increasing
        """
        g = nx.erdos_renyi_graph(1000, 0.01)
        dendo = co.generate_dendogram(g)
        mod_prec = -1.
        mods = [co.modularity(co.partition_at_level(dendo, level), g) for level in range(len(dendo)) ]
        self.assertListEqual(mods, sorted(mods))

    def test_nodes_stay_together(self):
        """
        Test that two nodes in the same community at one level stay in the same at higher level
        """
        g = nx.erdos_renyi_graph(500, 0.01)
        dendo = co.generate_dendogram(g)
        parts = dict([])
        for l in range(len(dendo)) :
            parts[l] = co.partition_at_level(dendo, l)
        for l in range(len(dendo)-1) :
            p1 = parts[l]
            p2 = parts[l+1]
            coms = set(p1.values())
            for com in coms :
                comhigher = [ p2[node] for node, comnode in p1.iteritems() if comnode == com]
                self.assertEqual(len(set(comhigher)), 1)

        
if __name__ == '__main__':
    unittest.main()