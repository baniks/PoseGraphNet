import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import cv2
import numpy as np

import matplotlib.pyplot as plt
import itertools

import numpy as np
from scipy.special import softmax

def plot_adjacency_matrix(adj_group, plot_name="../out/plot_adjacency.jpg"):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    adj_group: 3 x 17 x 17 or 1 x 17 x 17

    """

    adj_group = np.around(adj_group, decimals=2)

    joint_names = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck','Head','LShoulder', 'LElbow','LWrist','RShoulder','RElbow','RWrist']

    figure, axs = plt.subplots(1, adj_group.shape[0], figsize=(20,20))

    if adj_group.shape[0] == 1:
        axs = [axs]
    
    for group_id in range(adj_group.shape[0]):
        ax = axs[group_id]
        adj = adj_group[group_id]
        im = ax.imshow(adj, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.Blues)
        figure.colorbar(im, ax=ax, shrink=0.75)
        tick_marks = np.arange(len(joint_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(joint_names, rotation=45, fontsize=20, horizontalalignment="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(joint_names, fontsize=20)
        ax.set_ylim(len(adj)-0.5, -0.5)
        
        # Use white text if squares are dark; otherwise black.
        threshold = 0.5
        for i, j in itertools.product(range(adj.shape[0]), range(adj.shape[1])):
            color = "white" if adj[i, j] > threshold else "black"
            ax.text(j, i, adj[i, j], horizontalalignment="center", color=color, fontsize=15)

    plt.savefig(plot_name, bbox_inches="tight")
    return 


# adj_g = np.random.rand(1, 17, 17)
# plot_adjacency_matrix(adj_g)
