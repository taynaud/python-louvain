#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
__all__ = ["partition_at_level", "modularity", "best_partition", "generate_dendogram", "induced_graph"]
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001

import networkx as nx
import sys
import types
import array


def partition_at_level(dendogram, level) :
    """Return the partition of the nodes at the given level

    Level 0 is the first partition, and the best is len(dendogram) - 1

    :param dendogram: a list of partitions, ie dictionnaries where keys of the i+1 are the values of the i.
    :type dendogram: list of dictionary
    :param level: an integer which belongs to [0..len(dendogram)-1]
    :type level: integer
    :rtype: dictionary
    :return: a dictionary where keys are the nodes and the values are the set it belongs to

    """
    partition = dendogram[0].copy()
    for index in range(1, level + 1) :
        for node, community in partition.iteritems() :
            partition[node] = dendogram[index][community]
    return partition
    

def modularity(partition, graph) :
    """Compute the modularity of a partition of a graph

    :param partition: the partition of the nodes, i.e a dictionary where keys are their nodes and values the communities
    :type partition: dictionary
    :param graph: the networkx graph which is decomposed
    :type graph: networkx graph
    :rtype: float
    :return: The modularity

    """
    if type(graph) != nx.Graph :
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weighted = True)
    if links == 0 :
        raise ValueError("A graph without link has an undefined modularity")
    
    for node in graph :
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weighted = True)
        for neighbor, datas in graph[node].iteritems() :
            weight = datas.get("weight", 1)
            if partition[neighbor] == com :
                if neighbor == node :
                    inc[com] = inc.get(com, 0.) + float(weight)
                else :
                    inc[com] = inc.get(com, 0.) + float(weight) / 2.

    res = 0.
    for com in set(partition.values()) :
        res += (inc.get(com, 0.) / links) - (deg.get(com, 0.) / (2.*links))**2
    return res


def best_partition(graph, partition = None) :
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices


    
    :param graph: the networkx graph which is decomposed
    :type graph: networkx graph
    :param partition: the algorithm will start using this partition of the nodes. It's a dictionary where keys are their nodes and values the communities
    :type partition: dictionary, optional
    :rtype: dictionary
    :return: The partition, with communities numbered from 0 to number of communities

    """
    dendo = generate_dendogram(graph, partition)
    return partition_at_level(dendo, len(dendo) - 1 )


def generate_dendogram(graph, part_init = None) :
    """Find communities in the graph and return the associated dendogram

    :param graph: the networkx graph which will be decomposed
    :type graph: networkx graph
    :param part_init: the algorithm will start using this partition of the nodes. It's a dictionary where keys are their nodes and values the communities
    :type part_init: dictionary, optional
    :rtype: list of dictionaries
    :return: a list of partitions, ie dictionnaries where keys of the i+1 are the values of the i. and where keys of the first are the nodes of graph

    """
    if type(graph) != nx.Graph :
        raise TypeError("Bad graph type, use only non directed graph")
    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, part_init)
    mod = __modularity(status)
    status_list = list()
    __one_level(current_graph, status)
    new_mod = __modularity(status)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph)
    status.init(current_graph)
    
    while True :
        __one_level(current_graph, status)
        new_mod = __modularity(status)
        if new_mod - mod < __MIN :
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph)
        status.init(current_graph)
    return status_list[:]


def induced_graph(partition, graph) :
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    :param partition: a dictionary where keys are graph nodes and  values the part the node belongs to
    :type partition: dictionary
    :param graph: the initial graph
    :type graph: networkx graph
    :rtype: networkx.Graph
    :return: a networkx graph where nodes are the parts

    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())
    
    for node1, node2, datas in graph.edges_iter(data = True) :
        weight = datas.get("weight", 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {"weight":0}).get("weight", 1)
        ret.add_edge(com1, com2, weight = w_prec + weight)
        
    return ret


def __renumber(dictionary) :
    """Renumber the values of the dictionary from 0 to n

    :param dictionary: the partition
    :type dictionary: dictionary
    :rtype: dictionary
    :return: The modified partition

    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])
    
    for key in dictionary.keys() :
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1 :
            new_values[value] = count
            new_value = count
            count = count + 1
        ret[key] = new_value
        
    return ret


def __load_binary(data) :
    """Load binary graph as used by the cpp implementation of this algorithm

    :param data: the file containing the data
    :type data: string or file
    :rtype: networkx.Graph
    :return: The graph

    """
    if type(data) == types.StringType :
        data = open(data, "rb")
        
    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0
    
    for index in range(num_nodes) :
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg
        
    return graph


def __one_level(graph, status) :
    """Compute one level of communities

    :param graph: the graph we are working on
    :type graph: dictionary
    :param status: a named tuple with node2com, total_weight, internals, degrees set
    :type status: Status
    :return: nothing, the status is modified during the function

    """
    modif = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod
    
    while modif  and nb_pass_done != __PASS_MAX :
        cur_mod = new_mod
        modif = False
        nb_pass_done += 1
        
        for node in graph.nodes() :
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight*2.)
            neigh_communities = __neighcom(node, graph, status)
            __remove(node, com_node,
                    neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.iteritems() :
                incr =  dnc  - status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase :
                    best_increase = incr
                    best_com = com                    
            __insert(node, best_com,
                    neigh_communities.get(best_com, 0.), status)
            if best_com != com_node :
                modif = True                
        new_mod = __modularity(status)
        if new_mod - cur_mod < __MIN :
            break


class Status :
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}
    
    def __init__(self) :
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])
        
    def __str__(self) :
        return ("node2com : " + str(self.node2com) + " degrees : "
            + str(self.degrees) + " internals : " + str(self.internals)
            + " total_weight : " + str(self.total_weight))

    def copy(self) :
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, part = None) :
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weighted = True)
        if part == None :
            for node in graph.nodes() :
                self.node2com[node] = count
                deg = float(graph.degree(node, weighted = True))
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                self.loops[node] = float(graph.get_edge_data(node, node,
                                                 {"weight":0}).get("weight", 1))
                self.internals[count] = self.loops[node]
                count = count + 1
        else :
            for node in graph.nodes() :
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weighted = True))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].iteritems() :
                    weight = datas.get("weight", 1)
                    if part[neighbor] == com :
                        if neighbor == node :
                            inc += float(weight)
                        else :
                            inc += float(weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc



def __neighcom(node, graph, status) :
    """
    Compute the communities in the neighborood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].iteritems() :
        if neighbor != node :
            weight = datas.get("weight", 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + weight
            
    return weights


def __remove(node, com, weight, status) :
    """ Remove node from community com and modify status"""
    status.degrees[com] = ( status.degrees.get(com, 0.)
                                    - status.gdegrees.get(node, 0.) )
    status.internals[com] = float( status.internals.get(com, 0.) -
                weight - status.loops.get(node, 0.) )
    status.node2com[node] = -1
    

def __insert(node, com, weight, status) :
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = ( status.degrees.get(com, 0.) +
                                status.gdegrees.get(node, 0.) )
    status.internals[com] = float( status.internals.get(com, 0.) +
                        weight + status.loops.get(node, 0.) )


def __modularity(status) :
    """
    Compute the modularity of the partition of the graph
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()) :
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0 :
            result = result + in_degree / links - ((degree / (2.*links))**2)
    return result


def __main() :
    """Main function"""
    try :
        filename = sys.argv[1]
        graphfile = __load_binary(filename)
        partition = best_partition(graphfile)
        print >> sys.stderr, str(modularity(partition, graphfile))
        for elem, part in partition.iteritems() :
            print str(elem) + " " + str(part)
    except (IndexError, IOError):
        print "Usage : ./community filename"
        print "find the communities in graph filename and display the dendogram"
        print "Parameters:"
        print "filename is a binary file as generated by the "
        print "convert utility distributed with the C implementation"

    

if __name__ == "__main__" :
    __main()


