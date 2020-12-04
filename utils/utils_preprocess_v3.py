import pandas as pd
import numpy as np
import sys 
import os
import itertools

import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from scipy.sparse import csr_matrix, vstack, hstack
from copy import deepcopy
module_path = os.path.abspath(os.path.join('..'))


def getReducedGraph(sample_nodes, graph_nodes, 
					interactome):

	'''
	Reduce graph with only intersection nodes from sample and
	interactome. 
	'''

	#find intersection between sample nodes and graph nodes
	sample_nodes = set(sample_nodes)
	graph_nodes = set(graph_nodes)
	intersection_nodes = sample_nodes.intersection(graph_nodes)
	print('Number of Intersection Nodes : ', len(intersection_nodes))

	g = []
	for line in tqdm_notebook(range(len(interactome))):
		if (interactome.iloc[line]['node1'] in intersection_nodes
			and interactome.iloc[line]['node2'] in intersection_nodes):
			g.append(interactome.iloc[line])
	
	return pd.DataFrame(g)


def getNodeCharacterization(g, sample_nodes):
	'''
	Characterizes nodes based on if node is connected or orphan
	'''
	connected_nodes = set(g.nodes())
	orphan_nodes = set(sample_nodes) - connected_nodes
	
	return connected_nodes, orphan_nodes

def getDataSorting(connected_nodes, sample_df):
	'''
	Sorts covariant matrix such that connected nodes are first
	followed by orphan nodes and nodes not in interactome
	'''
	sample_df_sorted = deepcopy(sample_df)
	sample_df_sorted['IN_INTERACTOME'] = sample_df["node"].isin(list(connected_nodes)).tolist()
	sample_df_sorted = sample_df_sorted.sort_values(by="IN_INTERACTOME", ascending=False).reset_index(drop=True)
	
	#get dictionary to map node to number
	num_to_node = {}
	for i,nod in enumerate(sample_df_sorted['node'].tolist()):
		num_to_node[i] = nod
	
	#get ordered list of nodes in interactome
	ordered_nodelist = sample_df_sorted.loc[sample_df_sorted['IN_INTERACTOME'] == True]['node'].tolist()
	
	#delete 'IN_INTERACTOME' column
	sample_df_sorted = sample_df_sorted.drop(columns = ['IN_INTERACTOME', 'node'])
	
	return sample_df_sorted, ordered_nodelist, num_to_node

def getLaplacian(g, ordered_nodelist, orphan_nodes):
	'''
	Calculates laplacian matrix with respect to ordering of 
	covariant matrix
	'''
	L_norm = nx.normalized_laplacian_matrix(g, nodelist = ordered_nodelist, weight = 'confidence')
	L = nx.laplacian_matrix(g, nodelist = ordered_nodelist, weight = 'confidence')
	return csr_matrix(scipy.linalg.block_diag(L.todense(),np.eye(len(orphan_nodes)))), \
		csr_matrix(scipy.linalg.block_diag(L_norm.todense(),np.eye(len(orphan_nodes))))


class Preprocessing():
	
	def __init__(self):
		self.g = None
		self.connected_nodes = None
		self.orphan_nodes = None
		self.sorted_X = None
		self.ordered_nodelist = None
		self.num_to_node = None
		self.L = None
		self.L_norm = None
		
	def transform(self,X_nodes, graph_nodes, graph, X, save_location, load_graph = False):
		
		if load_graph == False:
			self.g = getReducedGraph(X_nodes, graph_nodes, graph)
			self.g.to_csv(save_location, header=None, index=None, sep='\t')
		
		self.g = nx.read_edgelist(save_location, 
					 data=(('confidence',float),))
		
		self.connected_nodes, self.orphan_nodes = \
			getNodeCharacterization(self.g, X_nodes)
		
		self.sorted_X, self.ordered_nodelist, self.num_to_node = \
			getDataSorting(self.connected_nodes,X)
		
		self.L, self.L_norm = getLaplacian(self.g, self.ordered_nodelist, self.orphan_nodes, )
		