import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def getTranslatedNodes(features, coefs, num_to_node, g, save_location = None):
    
    up_features = [num_to_node[features[i]] for i in range(len(features)) if coefs[i] > 0 ]
    down_features = [num_to_node[features[i]] for i in range(len(features)) if coefs[i] < 0 ]
    selected_features = [num_to_node[i] for i in features]
    
    sub_graph = nx.Graph(g.subgraph(selected_features))
    for nd in selected_features:
        if nd not in sub_graph.nodes():
            sub_graph.add_node(nd)        
    sub_graph = nx.freeze(sub_graph)

    pos = nx.circular_layout(sub_graph)
    
    
    nx.draw_networkx_nodes(sub_graph, pos = pos, node_color = 'b',
                           nodelist = up_features, 
                          alpha = 0.3)
    
        
    nx.draw_networkx_nodes(sub_graph, pos = pos, node_color = 'r',
                           nodelist = down_features, 
                          alpha = 0.3)
    
    nx.draw_networkx_edges(sub_graph, pos, width = 1)
    nx.draw_networkx_labels(sub_graph, pos, font_size=10, font_family='sans-serif', font_weight='bold')
    
    xmax= max(xx for xx,yy in pos.values()) + 0.2
    ymax= max(yy for xx,yy in pos.values()) + 0.2
    xmin= min(xx for xx,yy in pos.values()) - 0.2
    ymin= min(yy for xx,yy in pos.values()) - 0.2
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.box(False)
    if save_location:
        plt.savefig(save_location, dpi = 300)
    plt.show()

def getGridsearchPlot(gridsearch_results, alpha_list, threshold_list, save_location = None):
    grid = np.array(gridsearch_results['Test MSE']).reshape(len(alpha_list), len(threshold_list))
    sns.heatmap(np.flipud(grid), xticklabels=["{0:.3f}".format(i) for i in threshold_list], 
                    yticklabels=np.flip(["{0:.2f}".format(i) for i in alpha_list]), vmin = 0.0, vmax = 1.0)
    plt.yticks(rotation = 0, fontsize=12)
    plt.xticks(rotation = 90, fontsize=12)
    plt.xlabel(r'$\tau$', fontsize = 16)
    plt.ylabel(r'$\lambda$', fontsize = 16)
    if save_location:
        plt.savefig(save_location, dpi = 300)
    plt.show()