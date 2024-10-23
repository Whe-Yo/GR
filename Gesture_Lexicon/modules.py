import numpy as np
import torch
import torch.nn as nn

# class NormalizeBlock(nn.Module):
#     def __init__(self, beat, text):
#         super().__init__()
#
#         self.

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride):
        super().__init__()

        self.stgcn1 = StgcnBlock(dim_in=dim_in, dim_out=dim_in, kernel_size=kernel_size, padding=padding, stride=stride)
        self.stgcn2 = StgcnBlock(dim_in=dim_in, dim_out=dim_out, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x, adj):
        out = self.conv(x)
        x = self.stgcn1(x, adj)
        x = self.stgcn2(x, adj)
        out = out + 0.1 * x

        return out
class StgcnBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride):
        super().__init__()
        self.gcn = SpatialConv(dim_in, dim_out, kernel_size[1])
        self.tcn = nn.Conv2d(dim_out, dim_out, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=stride)

        self.lRelu = nn.LeakyReLU()

    def forward(self, x, adj):
        x = self.lRelu(x)

        x = self.gcn(x, adj)
        x = self.tcn(x)

        return x

class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
                              padding_mode='reflect', stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, adj):
        assert adj.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj))

        return x.contiguous()

class SpatialDownsample(nn.Module):
    def __init__(self, hierarchy_index):
        super().__init__()
        self.device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        self.down_sample_graph = torch.Tensor(self.make_graph(hierarchy_index)).to(self.device)

    def make_graph(self, index):
        graph = np.zeros((16 // (2 ** index), 16 // (2 ** (index  + 1))))
        for i in range(0, 16 // (index + 1), 2):
            graph[i][i // 2] = 1 / 2
            graph[i + 1][i // 2] = 1 / 2
        return graph

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.down_sample_graph)
        return x.contiguous()

class SpatialUpsample(nn.Module):
    def __init__(self, hierarchy_index):
        super().__init__()
        self.device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        self.up_sample_graph = torch.Tensor(self.make_graph(hierarchy_index)).to(self.device)

    def make_graph(self, index):
        graph = np.zeros((16 // (2 ** (2 - index)), 16 // (2 ** (1 - index))))
        for i in range(16 // (2 ** (2 - index))):
            graph[i][i * 2] = 1
            graph[i][i * 2 + 1] = 1
        return graph

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.up_sample_graph)
        return x.contiguous()
class GraphJoint():
    def __init__(self, hierarchy_index):
        self.num_node = [16, 8, 4]
        self.hop = [2, 1, 1]
        self.adj_mat = self.get_adj_mat(hierarchy_index)

    def get_adj_mat(self, idx):
        # save edge information
        self_link = [(i, i) for i in range(self.num_node[idx])]
        neighbor_link = [[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                         (4, 8), (8, 9), (9, 10), (10, 11),
                         (4, 12), (12, 13), (13, 14), (14, 15)],    # idx == 0
                         [(0, 1), (1, 2), (2, 3),
                          (2, 4), (4, 5),
                          (2, 6), (6, 7)],                          # idx == 1
                         [(0, 1), (1, 2), (1, 3)]]                  # idx == 2
        link = self_link + neighbor_link[idx]

        # calc distance
        adj_mat = np.zeros((self.num_node[idx], self.num_node[idx]))
        for i, j in link:
            adj_mat[i, j] = 1
            adj_mat[j, i] = 1

        hop_dist = np.zeros((self.num_node[idx], self.num_node[idx])) + np.inf
        transfer_matrix = [np.linalg.matrix_power(adj_mat, d) for d in range(self.hop[idx] + 1)]
        arrive_matrix = (np.stack(transfer_matrix) > 0)
        # 2에서 시작 0 까지 -1마다
        for dist in range(self.hop[idx], -1, -1):
            hop_dist[arrive_matrix[dist]] = dist

        # calc adjacency matrix
        valid_hop = range(0, self.hop[idx] + 1)
        adj_mat = np.zeros((self.num_node[idx], self.num_node[idx]))
        for hop in valid_hop:
            adj_mat[hop_dist == hop] = 1

        # calc normalized adjacency matrix
        mat_sum = np.sum(adj_mat, axis=0)
        weight_mat = np.zeros(adj_mat.shape)
        for i in range(adj_mat.shape[0]):
            if mat_sum[i] > 0:
                weight_mat[i, i] = mat_sum[i] ** (-1)
        norm_adj_mat = np.dot(adj_mat, weight_mat)

        adj_mat = np.zeros((len(valid_hop), self.num_node[idx], self.num_node[idx]))
        for i, hop in enumerate(valid_hop):
            adj_mat[i][hop_dist == hop] = norm_adj_mat[hop_dist == hop]

        return adj_mat # (3, 16, 16)
