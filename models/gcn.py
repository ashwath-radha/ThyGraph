import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
# from torch_geometric.nn import SAGPooling
import torchvision.models as models
from torch import Tensor
from torch_geometric.utils import scatter
from typing import Optional

from typing import Callable, Optional, Tuple, Union
import os
import sys
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from utils.utils import initialize_gnn_weights, initialize_weights

# GAT + Global Mean Pooling
class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads=8):
        print('Creating GAT module')
        super(GAT, self).__init__()
        
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim*heads, output_dim, heads=1)
        self.dropout = nn.Dropout(p=0.5)   # Dropout layer with probability 0.5
        self.fc = nn.Linear(output_dim, 2)  # Final layer for binary classification

        # Initialize weights
        initialize_gnn_weights(self)
        initialize_weights(self)

    def forward(self, input_data: torch.Tensor, return_attention: bool = False, return_features: bool = False):
        x = input_data['frame_features']['x'].squeeze()
        edge_index = input_data['frame_features']['edge_index']
        batch = input_data['frame_features']['batch'] # is this what it's supposed to look like

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x, A = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        logits = self.fc(x)
        y_prob = F.softmax(logits, dim=1)
        # print(y_prob[:,1])

        if return_features: # this is probably not in the right spot
            return x
        if return_attention:
            return A
        return logits, y_prob

# GCN + Global Mean Pooling
class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        print('Creating GCN module')
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(output_dim, 2)  # Final layer for binary classification

       # Initialize weights
        initialize_gnn_weights(self)
        initialize_weights(self)

    def forward(self, input_data: torch.Tensor):
        x = input_data['frame_features']['x'].squeeze()
        edge_index = input_data['frame_features']['edge_index']
        batch = input_data['frame_features']['batch']

        x = self.conv1(x, edge_index)
        x = F.relu(x) # change to F.leaky_relu(x, negative_slope=0.2) ?
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        logits = self.fc(x)
        y_prob = F.softmax(logits, dim = 1)

        return logits, y_prob

# GCN + SAGPooling
class SAG_GCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pooling_ratio=[0.75, 0.75], n_layers=3, gnn='GraphConv'):
        super(SAG_GCN, self).__init__()
        # print('manual parameters: ', hidden_dim, pooling_ratio, n_layers, gnn)
        self.pooling_ratio = [float(ratio) for ratio in pooling_ratio[0].split(' ')] #[0.5, 0.75]
        self.dropout_ratio = 0.5
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        if gnn=='GraphConv':
            self.gnn = GraphConv
        elif gnn=='GCNConv':
            self.gnn = GCNConv
        elif gnn=='GATConv':
            self.gnn = GATConv
        # print('manual parameters: ', self.hidden_dim, self.pooling_ratio, self.n_layers, self.gnn)
        
        self.conv_layers = torch.nn.ModuleList()
        self.pool_layers = torch.nn.ModuleList()
        assert self.n_layers==len(self.pooling_ratio)
        # Add conv1 and pool1 layers dynamically based on n_layers
        for i in range(n_layers):
            if i == 0:
                # First layer
                self.conv_layers.append(GCNConv(input_dim, self.hidden_dim))
                self.pool_layers.append(SAGPooling(self.hidden_dim, ratio=self.pooling_ratio[i], GNN=self.gnn))
            else:
                # Additional layers
                self.conv_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
                self.pool_layers.append(SAGPooling(self.hidden_dim, ratio=self.pooling_ratio[i], GNN=self.gnn))

        self.lin1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim//2)
        self.lin2 = torch.nn.Linear(self.hidden_dim//2, self.hidden_dim//8)
        self.lin3 = torch.nn.Linear(self.hidden_dim//8, 2)

        # Initialize weights
        initialize_gnn_weights(self)
        initialize_weights(self)

    def forward(self, input_data: torch.Tensor, device: str='cuda', return_attention: bool = False, return_features: bool = False):
        x = input_data['frame_features']['x'].squeeze()
        edge_index = input_data['frame_features']['edge_index']
        batch = input_data['frame_features']['batch']

        to_sum = []
        for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _, A = pool(x, edge_index, None, batch)
            if i==0 and return_attention:
                return_A = A
            x1 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)
            to_sum.append(x1)

        x = sum(to_sum)

        # x = F.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, A = self.pool1(x, edge_index, None, batch)
        # x1 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)

        # x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)

        # # x = F.relu(self.conv3(x, edge_index))
        # # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # # x3 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)

        # x = x1 + x2 # + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        logits = self.lin3(x)
        y_prob = F.softmax(logits, dim=-1)
        if return_attention:
            print(y_prob[:,1])
            return return_A

        return logits, y_prob

# # GCN + SAGPooling
# class SAG_GCN(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
#         super(SAG_GCN, self).__init__()
#         self.pooling_ratio = 0.75
#         self.dropout_ratio = 0.5
#         self.gnn = GraphConv # GraphConv, GATConv, GCNConv
        
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.pool1 = SAGPooling(hidden_dim, ratio=self.pooling_ratio, GNN=self.gnn)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.pool2 = SAGPooling(hidden_dim, ratio=self.pooling_ratio, GNN=self.gnn)

#         self.lin1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
#         self.lin3 = torch.nn.Linear(output_dim, 2)

#         # Initialize weights
#         initialize_gnn_weights(self)
#         initialize_weights(self)

#     def forward(self, input_data: torch.Tensor, device: str='cuda', return_attention: bool = False, return_features: bool = False):
#         x = input_data['frame_features']['x'].squeeze()
#         edge_index = input_data['frame_features']['edge_index']
#         batch = input_data['frame_features']['batch']

#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _, A = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)

#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)
        
#         x = x1 + x2

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin2(x))
#         logits = self.lin3(x)
#         y_prob = F.softmax(logits, dim=-1)
#         if return_attention:
#             print(y_prob[:,1])
#             return A

#         return logits, y_prob

# GAT + SAGPooling
class SAG_GAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads=8):
        super(SAG_GAT, self).__init__()
        self.pooling_ratio = 0.75
        self.dropout_ratio = 0.5
        self.heads = heads
        self.gnn = GATConv # GraphConv, GATConv, GCNConv

        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=self.heads)
        self.pool1 = SAGPooling(hidden_dim*self.heads, ratio=self.pooling_ratio, GNN=self.gnn)
        self.gat2 = GATv2Conv(hidden_dim*self.heads, hidden_dim, heads=1)
        self.pool2 = SAGPooling(hidden_dim, ratio=self.pooling_ratio, GNN=self.gnn)

        self.lin1 = torch.nn.Linear(hidden_dim*2*self.heads, hidden_dim*self.heads)
        self.lin2 = torch.nn.Linear(hidden_dim*self.heads, output_dim*(self.heads//2))
        self.lin3 = torch.nn.Linear(output_dim*(self.heads//2), output_dim*(self.heads//4))
        self.lin4 = torch.nn.Linear(output_dim*(self.heads//4), 2)

        # Initialize weights
        initialize_gnn_weights(self)
        initialize_weights(self)

    def forward(self, input_data: torch.Tensor, device: str='cuda', return_attention: bool = False, return_features: bool = False):
        x = input_data['frame_features']['x'].squeeze()
        edge_index = input_data['frame_features']['edge_index']
        batch = input_data['frame_features']['batch']

        x, A1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)
        
        x, A2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        gmaxp = torch.tile(global_max_pool(x.cpu(), batch.cpu()).to(device), (1, self.heads))
        gmeanp = torch.tile(global_mean_pool(x, batch), (1, self.heads))
        x2 = torch.cat([gmaxp, gmeanp], dim=1)

        if return_attention:
            return A2

        x = x1 + x2
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin3(x))
        logits = self.lin4(x)
        y_prob = F.softmax(logits, dim=-1)

        return logits, y_prob


# from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor

class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}.

    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (str or callable, optional): The non-linearity to use.
            (default: :obj:`"tanh"`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GCNConv, #GATConv, #GCNConv, #GraphConv, # replaced this
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.gnn = GNN(in_channels, 1, **kwargs)
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        # print("IN SAG POOL FORWARD")
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = self.gnn(attn, edge_index)
        # print(min(attn), max(attn))
        # print(len(attn), len(attn[0]), len(attn[1]))
        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr,
                connect_out.batch, perm, attn)
        # return (x, connect_out.edge_index, connect_out.edge_attr,
        #         connect_out.batch, perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')