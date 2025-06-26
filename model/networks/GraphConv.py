import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch_scatter import scatter_mean

class GraphConv(torch.nn.Module):

	def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7, n_node_type=0, use_bias=False):
		super().__init__()
		self.in_channels = in_channels#入力チャネル数
		self.out_channels = out_channels#出力チャネル数
		self.use_bias = use_bias
		self.n_edge_type = n_edge_type
		self.avg_degree = avg_degree
		self.n_node_type = n_node_type#ノードが属するOctreeの深さ

		node_channel: int = n_node_type if n_node_type > 1 else 0
		self.weights = torch.nn.Parameter(
			torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels))
		if self.use_bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))


		self.reset_parameters()

	def reset_parameters(self) -> None:
		fan_in = self.avg_degree * self.in_channels
		fan_out = self.avg_degree * self.out_channels
		std = math.sqrt(2.0 / float(fan_in + fan_out))
		a = math.sqrt(3.0) * std
		torch.nn.init.uniform_(self.weights, -a, a)
		if self.use_bias:
			torch.nn.init.zeros_(self.bias)

	def forward(self, x, doctree, d):
		edge_index = doctree.graph[d]['edge_idx']
		edge_type = doctree.graph[d]['edge_dir']
		node_type = doctree.graph[d]['node_type']
		has_node_type = node_type is not None
		if has_node_type and self.n_node_type > 1:
			# concatenate the one_hot vector
			one_hot = F.one_hot(node_type, num_classes=self.n_node_type)
			x = torch.cat([x, one_hot], dim=1)

		# x -> col_data
		row, col = edge_index[0], edge_index[1]
		# weights = torch.pow(0.5, node_type[col]) if has_node_type else None
		weights = None    # TODO: ablation the weights
		index = row * self.n_edge_type + edge_type
		col_data = scatter_mean(x[col], index, dim=0, weights=weights,
			dim_size=x.shape[0] * self.n_edge_type)

		# matrix product
		output = col_data.view(x.shape[0], -1) @ self.weights

		

		if self.use_bias:
			output += self.bias

		return output
	
	def forward_test(self, x, doctree, d):
		edge_index = doctree.graph[d]['edge_idx']
		edge_type = doctree.graph[d]['edge_dir']
		node_type = doctree.graph[d]['node_type']
		has_node_type = node_type is not None
		if has_node_type and self.n_node_type > 1:
			# concatenate the one_hot vector
			one_hot = F.one_hot(node_type, num_classes=self.n_node_type)
			x = torch.cat([x, one_hot], dim=1)

		# x -> col_data
		row, col = edge_index[0], edge_index[1]
		# weights = torch.pow(0.5, node_type[col]) if has_node_type else None
		weights = None    # TODO: ablation the weights
		index = row * self.n_edge_type + edge_type
		col_data = scatter_mean(x[col], index, dim=0, weights=weights,
			dim_size=x.shape[0] * self.n_edge_type)

		# matrix product
		# if self.direct_method:
		# col_data.requires_grad = True
		output = col_data.view(x.shape[0], -1) @ self.weights
		# output.requires_grad = True

		# grad1 = torch.autograd.grad(output, [col_data], grad_outputs=torch.ones_like(output), create_graph=True)[0]
		# # else:
		# grad2 = MatrixProdOp.backward(None, col_data, self.weights, torch.ones_like(output))
		# torch.abs(grad1 - grad2).sum()
		

		if self.use_bias:
			output += self.bias

		return output

	def extra_repr(self) -> str:
		return ('channel_in={}, channel_out={}, n_edge_type={}, avg_degree={}, '
			'n_node_type={}'.format(self.in_channels, self.out_channels,
			self.n_edge_type, self.avg_degree, self.n_node_type))  # noqa