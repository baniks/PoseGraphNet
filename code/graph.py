'''adapted from
https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks'''

import numpy as np


class Graph():
    """ The Graph to model the skeletons of human body/hand
    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration
        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1,
                 with_hip=False,
                 norm=True):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout, with_hip)

        self.hop_dis = self.get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)

        
        self.dist_center = self.get_distance_to_center(layout, with_hip)
        self.get_adjacency(strategy, norm)

    def __str__(self):
        return self.A


    def get_hop_distance(self, num_node, edge, max_hop=1):
        """
            for self link hop_dis=0
            for one hop neighbor hop_dis=1
            rest hop_dis=inf
        """
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            # bidirectional
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def get_distance_to_center(self,layout, with_hip=False):
        """
        :return: get the distance of each node to center
        """

        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            if with_hip == False:
                # center = spine
                dist_center[0 : 6] = [1, 2, 3, 1, 2, 3] # legs
                dist_center[6 : 10] = [0, 1, 2, 3] # body
                dist_center[10 : 16] = [2, 3, 4, 2, 3, 4] #arms
            else:
                # center = hip
                dist_center[0 : 7] = [0, 1, 2, 3, 1, 2, 3] # legs
                dist_center[7 : 11] = [1, 2, 3, 4] # body
                dist_center[11 : 17] = [3, 4, 5, 3, 4, 5] #arms

        return dist_center

    def get_edge(self, layout, with_hip=False):
        """
        get edge link of the graph
        self link +  one hop neighbors
        cb: center bone
        """
        if layout == 'hm36_gt':
            if with_hip == False:
                self.num_node = 16
                self.self_link = [(i, i) for i in range(self.num_node)]

                self.neighbour_link = [ (6,0), (0,1), (1,2), # Rleg
                                (6,3), (3,4), (4,5), # Lleg
                                (6,7), (7,8), (8,9), # body
                                (7,10), (10,11), (11,12), # Larm
                                (7,13), (13,14), (14,15) # Rarm
                                ]
                self.sym_link = [
                   (0,3), (1,4), (2,5), # legs
                   (13, 10), (14, 11), (15, 12) # arms
                ]
            else:
                self.num_node = 17
                self.self_link = [(i, i) for i in range(self.num_node)]

                self.neighbour_link = [ (0,1), (1,2), (2,3), # Rleg
                                (0,4), (4,5), (5,6), # Lleg
                                (0,7), (7,8), (8,9), (9,10), # body
                                (8,11), (11,12), (12,13), # Larm
                                (8,14), (14,15), (15,16) # Rarm
                                ]
                self.sym_link = [
                   (1,4), (2,5), (3,6), # legs
                   (14, 11), (15, 12), (16,13) # arms
                ]
                self.la, self.ra =[11, 12, 13], [14, 15, 16]
                self.ll, self.rl = [4, 5, 6], [1, 2, 3]
                self.cb = [0, 7, 8, 9, 10]
                self.part = [self.la, self.ra, self.ll, self.rl, self.cb]
                
            self.edge = self.self_link + self.neighbour_link # + self.sym_link

            # center node of body/hand
            self.center = 6 # Spine
        elif layout == 'itop':
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            neighbour_link = [ (9,11), (11,13), # Rleg
                (10,12), (12,14),  # Lleg
                (8,9), (8,10), (1,8), (0,1), # body
                (1,2), (2,4), (4,6), # Rarm
                (1,3), (3,5), (5,7) # Larm
                ]
            self.edge = self_link + neighbour_link
        elif layout == "h36m_sub":
            self.num_node = 5
            self.self_link = [(i, i) for i in range(self.num_node)]

            self.neighbour_link = [(0, 4), (1, 4), (2, 4), (3, 4)]
            self.sym_link = [(0,1), (2,3)]
            
            self.edge = self.self_link + self.neighbour_link + self.sym_link
        else:
            raise ValueError("This Layout does not exist.")


    def get_adjacency(self, strategy, norm=True):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        # creates self link and connects one hop neighbors
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        if norm:
            # normalize_adjacency = normalize_digraph(adjacency)
            normalize_adjacency = normalize_undigraph(adjacency)
        else:
            normalize_adjacency = adjacency
            print("unnormed adjacency")

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (j,i) in self.sym_link or (i,j) in self.sym_link:
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:                    
                    A.append(a_close)
                    A.append(a_further)
                    # A.append(a_sym)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")



def normalize_digraph(A):
    print("----adjacency norm_digraph-------------")
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    print("----adjacency norm_Undigraph-------------")
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

