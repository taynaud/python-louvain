import unittest
import networkx as nx
import community as co

class ModularityTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_trivialcases(self):
        #it test that everyone in one community has a modularity of 0
        for i in range(100) :
            g = nx.erdos_renyi_graph(50, 0.1)
            part = dict([])
            for node in g :
                part[node] = 0
            self.assertEqual(co.modularity(part, g), 0)
    

if __name__ == '__main__':
    unittest.main()