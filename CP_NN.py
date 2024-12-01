import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
import torch.nn.init as init
import torch.nn.functional as F
import networkx as nx
from parameters import *

class CPLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, CP_mask, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs)) 
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.cp_mask = CP_mask

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float() * self.cp_mask.float(), self.bias.float())

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def generate_random_sparse_mask(in_features, out_features, sparsity=0.5):
    """
    Generate a random sparse mask for an MLP weight matrix.
    
    Parameters:
    - in_features: int, number of input features (rows)
    - out_features: int, number of output features (columns)
    - sparsity: float, fraction of elements that should be zero (default: 0.9)
    
    Returns:
    - mask: torch.Tensor of shape [in_features, out_features] with 0s and 1s
    """
    # Calculate total elements and the number of non-zero elements
    total_elements = in_features * out_features
    num_nonzero_elements = int((1 - sparsity) * total_elements)
    
    # Create a mask filled with zeros
    mask = torch.zeros(in_features, out_features)
    
    # Randomly choose indices for the non-zero elements
    indices = torch.randperm(total_elements)[:num_nonzero_elements]
    
    # Set the chosen indices to 1
    mask.view(-1)[indices] = 1
    
    return mask

class RandomSparseLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs)) 
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.random_mask = generate_random_sparse_mask(in_features, out_features, sparsity=0.5)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float() * self.random_mask.float().to(device), self.bias.float())

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CP_NN(nn.Module):
    # 在上面的simpleNet网络的基础上，在每层的输出部分添加激励函数
    def __init__(self, in_dim, out_dim, CP_mask, device):
        super(CP_NN, self).__init__()
        #cp_linear = CPLinear(in_dim, out_dim, CP_mask, device=device)
        # 下部分代码（除了输出层）的输出部分都添加了激励函数，最后还用了nn.Sequential()函数，这个函数将
        # nn.Linear（）函数和nn.ReLU()函数组合到一起作为self。layer。注意输出层不能有激励函数，因为输出结果表示实际的预测值
        self.layer1 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer2 = nn.Sequential(nn.Linear(in_dim, out_dim))    #normal linear
        #self.layer2 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer3 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
    def forward(self, x):
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        return x


class RandomSparse_NN(nn.Module):
    # 在上面的simpleNet网络的基础上，在每层的输出部分添加激励函数
    def __init__(self, in_dim, out_dim, device):
        super(RandomSparse_NN, self).__init__()
        #cp_linear = CPLinear(in_dim, out_dim, CP_mask, device=device)
        # 下部分代码（除了输出层）的输出部分都添加了激励函数，最后还用了nn.Sequential()函数，这个函数将
        # nn.Linear（）函数和nn.ReLU()函数组合到一起作为self。layer。注意输出层不能有激励函数，因为输出结果表示实际的预测值
        self.layer1 = nn.Sequential(RandomSparseLinear(in_dim, out_dim, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer2 = nn.Sequential(nn.Linear(in_dim, out_dim))    #normal linear
        #self.layer2 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer3 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
    def forward(self, x):
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        return x



def generate_watts_strogatz_sparse_mask(in_features, out_features, k, p):
    """
    Generate a Watts-Strogatz small-world sparse mask for an MLP weight matrix.
    
    Parameters:
    - in_features: int, number of input features (rows)
    - out_features: int, number of output features (columns)
    - k: int, each node is connected to k nearest neighbors in ring topology
    - p: float, the probability of rewiring each edge
    
    Returns:
    - mask: torch.Tensor of shape [in_features, out_features] with 0s and 1s
    """
    # Create a Watts-Strogatz graph
    ws_graph = nx.watts_strogatz_graph(in_features, k, p)
    
    # Initialize a zero mask
    mask = torch.zeros(in_features, out_features)
    
    # Convert the graph adjacency matrix to a sparse mask
    for i in range(in_features):
        neighbors = list(ws_graph.neighbors(i))
        for neighbor in neighbors:
            if neighbor < out_features:  # Only consider neighbors within out_features range
                mask[i, neighbor] = 1
    
    return mask

class WSSparseLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs)) 
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.ws_mask = generate_watts_strogatz_sparse_mask(in_features, out_features, k = 4, p = 0.5 )
         # Number of nearest neighbors   # Rewiring probability
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float() * self.ws_mask.float().to(device), self.bias.float())

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class WSSparse_NN(nn.Module):
    # 在上面的simpleNet网络的基础上，在每层的输出部分添加激励函数
    def __init__(self, in_dim, out_dim, device):
        super(WSSparse_NN, self).__init__()
        #cp_linear = CPLinear(in_dim, out_dim, CP_mask, device=device)
        # 下部分代码（除了输出层）的输出部分都添加了激励函数，最后还用了nn.Sequential()函数，这个函数将
        # nn.Linear（）函数和nn.ReLU()函数组合到一起作为self。layer。注意输出层不能有激励函数，因为输出结果表示实际的预测值
        self.layer1 = nn.Sequential(WSSparseLinear(in_dim, out_dim, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer2 = nn.Sequential(nn.Linear(in_dim, out_dim))    #normal linear
        #self.layer2 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
        #self.layer3 = nn.Sequential(CPLinear(in_dim, out_dim, CP_mask, device=device), QuickGELU())   #cp_linear. nn.ReLU() or QuickGELU()
    def forward(self, x):
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        return x