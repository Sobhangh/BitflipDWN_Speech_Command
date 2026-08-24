import torch
import torch.nn.functional as F

from .lut_layer import LUTLayer


class DWNConvLayer(torch.nn.Module):
	def __init__(
		self,
		in_channels,
		depth=3,
		lut_rank=6,
		kernels=32,
		receptive_field=15,
		stride=1,
		channels_per_group=5,
		padding=0,
		ste=True,
		flatten_output=True,
		random_kernel_groups=False,
		learnable_connections=False,
		mapping_tau=0.01,
		debug=False,
	):
		super().__init__()

	
		assert in_channels > 0
		assert depth > 0
		assert lut_rank > 0
		assert kernels > 0
		assert receptive_field > 0
		assert stride > 0
		assert channels_per_group > 0
		assert in_channels % channels_per_group == 0
		assert isinstance(padding, (int, tuple))
		assert isinstance(ste, bool)
		assert isinstance(flatten_output, bool)
		assert isinstance(random_kernel_groups, bool)
		assert isinstance(learnable_connections, bool)
		assert isinstance(debug, bool)
		assert mapping_tau > 0
		assert in_channels % channels_per_group == 0, "in_channels must be divisible by channels_per_group."
		assert lut_rank ** depth <= receptive_field ** 2, "lut_rank^depth must be <= receptive_field^2 to fit in the convolution window."


		self.in_channels = int(in_channels)
		self.depth = int(depth)
		self.lut_rank = int(lut_rank)
		self.kernels = int(kernels)
		self.receptive_field = int(receptive_field)
		self.stride = int(stride)
		self.channels_per_group = int(channels_per_group)
		self.padding = self._to_pair(padding, "padding")
		self.ste = ste
		self.flatten_output = flatten_output
		self.random_kernel_groups = random_kernel_groups
		self.learnable_connections = learnable_connections
		self.mapping_tau = float(mapping_tau)
		self.debug = debug
        
		self.leaf_size = self.lut_rank ** self.depth
		self.level_node_counts = [self.lut_rank ** (self.depth - level - 1) for level in range(self.depth)]
        
		# Tree is represented as a sequence of LUT layers: one LUTLayer per tree level.
		self.lut_layers = torch.nn.ModuleList()
		prev_nodes_per_kernel = self.leaf_size
		for nodes_per_kernel in self.level_node_counts:
			mapping = self._build_level_mapping(prev_nodes_per_kernel, nodes_per_kernel)
			self.lut_layers.append(
				LUTLayer(
					input_size=self.kernels * prev_nodes_per_kernel,
					output_size=self.kernels * nodes_per_kernel,
					n=self.lut_rank,
					mapping=mapping,
					ste=self.ste,
				)
			)
			prev_nodes_per_kernel = nodes_per_kernel

		self.num_groups = self.in_channels // self.channels_per_group

		# Kernel-to-group assignment remains fixed, but connection mappings are learnable.
		self.register_buffer("kernel_groups", torch.empty(self.kernels, dtype=torch.int64), persistent=True)
		if self.learnable_connections:
			self.Cc = torch.nn.Parameter(
				torch.empty(self.kernels, self.leaf_size, self.channels_per_group, dtype=torch.float32),
				requires_grad=True,
			)
			self.Ch = torch.nn.Parameter(
				torch.empty(self.kernels, self.leaf_size, self.receptive_field, dtype=torch.float32),
				requires_grad=True,
			)
			self.Cw = torch.nn.Parameter(
				torch.empty(self.kernels, self.leaf_size, self.receptive_field, dtype=torch.float32),
				requires_grad=True,
			)
		else:
			self.register_buffer("Cc", torch.empty(self.kernels, self.leaf_size, dtype=torch.int64), persistent=True)
			self.register_buffer("Ch", torch.empty(self.kernels, self.leaf_size, dtype=torch.int64), persistent=True)
			self.register_buffer("Cw", torch.empty(self.kernels, self.leaf_size, dtype=torch.int64), persistent=True)

		self.reset_parameters()

	@staticmethod
	def _to_pair(value, name):
		if isinstance(value, int):
			return (value, value)
		if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
			return value
		raise ValueError(f"{name} must be an int or a tuple of 2 ints.")

	def _debug(self, msg):
		if self.debug:
			print(f"[DWNConvLayer] {msg}")

	def _build_level_mapping(self, prev_nodes_per_kernel, nodes_per_kernel):
		self._debug(
			f"build_level_mapping prev_nodes_per_kernel={prev_nodes_per_kernel}, nodes_per_kernel={nodes_per_kernel}"
		)
		mapping = torch.empty(self.kernels * nodes_per_kernel, self.lut_rank, dtype=torch.int32)
		for kernel_idx in range(self.kernels):
			kernel_input_base = kernel_idx * prev_nodes_per_kernel
			kernel_output_base = kernel_idx * nodes_per_kernel
			for node_idx in range(nodes_per_kernel):
				child_start = node_idx * self.lut_rank
				mapping[kernel_output_base + node_idx] = torch.arange(
					kernel_input_base + child_start,
					kernel_input_base + child_start + self.lut_rank,
					dtype=torch.int32,
				)
		return mapping

	def reset_parameters(self):
		self._debug("reset_parameters start")
		for lut_layer in self.lut_layers:
			with torch.no_grad():
				num_tables, table_size = lut_layer.luts.shape
				indices = torch.arange(table_size, device=lut_layer.luts.device, dtype=torch.int64)
				pass_through = ((indices & 1).float() * 2.0 - 1.0).unsqueeze(0)
				random_luts = torch.rand_like(lut_layer.luts) * 2.0 - 1.0
				use_pass_through = torch.rand(num_tables, 1, device=lut_layer.luts.device) < 0.9 #0.06
				lut_layer.luts.copy_(torch.where(use_pass_through, pass_through, random_luts))
		self.regenerate_connections()
		self._debug("reset_parameters done")

	def _init_onehot_logits(self, logits, indices):
		# Build one-hot-like logits so argmax starts from sampled discrete connections.
		logits.fill_(-4.0)
		logits.scatter_(-1, indices.unsqueeze(-1), 4.0)

	def regenerate_connections(self):
		self._debug("regenerate_connections start")
		if self.random_kernel_groups:
			kernel_groups = torch.randint(0, self.num_groups, (self.kernels,), dtype=torch.int64)
		else:
			kernel_groups = torch.arange(self.kernels, dtype=torch.int64) % self.num_groups

		self.kernel_groups.copy_(kernel_groups)

		if self.learnable_connections:
			with torch.no_grad():
				cc_idx = torch.randint(0, self.channels_per_group, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Cc.device)
				ch_idx = torch.randint(0, self.receptive_field, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Ch.device)
				cw_idx = torch.randint(0, self.receptive_field, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Cw.device)
				self._init_onehot_logits(self.Cc, cc_idx)
				self._init_onehot_logits(self.Ch, ch_idx)
				self._init_onehot_logits(self.Cw, cw_idx)
		else:
			self.Cc.copy_(torch.randint(0, self.channels_per_group, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Cc.device))
			self.Ch.copy_(torch.randint(0, self.receptive_field, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Ch.device))
			self.Cw.copy_(torch.randint(0, self.receptive_field, (self.kernels, self.leaf_size), dtype=torch.int64, device=self.Cw.device))
		self._debug(
			f"regenerate_connections done kernel_groups_shape={tuple(self.kernel_groups.shape)}, "
			f"learnable_connections={self.learnable_connections}, "
			f"Cc_shape={tuple(self.Cc.shape)}, Ch_shape={tuple(self.Ch.shape)}, Cw_shape={tuple(self.Cw.shape)}"
		)

	def _st_select(self, logits):
		if not self.learnable_connections:
			raise RuntimeError("_st_select called in random-connections mode.")
		self._debug(f"st_select logits_shape={tuple(logits.shape)}")
		probs = torch.softmax(logits / self.mapping_tau, dim=-1)
		indices = probs.argmax(dim=-1)
		hard = torch.zeros_like(probs).scatter_(-1, indices.unsqueeze(-1), 1.0)
		self._debug(
			f"st_select probs_shape={tuple(probs.shape)}, hard_shape={tuple(hard.shape)}, indices_shape={tuple(indices.shape)}"
		)
		return hard + probs - probs.detach()

	def hard_connection_indices(self):
		if self.learnable_connections:
			cc_local = self.Cc.argmax(dim=-1)
			ch = self.Ch.argmax(dim=-1)
			cw = self.Cw.argmax(dim=-1)
		else:
			cc_local = self.Cc
			ch = self.Ch
			cw = self.Cw

		group_offsets = (self.kernel_groups * self.channels_per_group).unsqueeze(1).to(cc_local.device)
		cc = cc_local + group_offsets
		return cc, ch, cw

	def _forward_random_connections(self, x, batch_size, out_h, out_w, height, width):
		cc, ch, cw = self.hard_connection_indices()
		self._debug(
			f"forward random mappings cc={tuple(cc.shape)}, ch={tuple(ch.shape)}, cw={tuple(cw.shape)}"
		)

		y_offsets = torch.arange(out_h, device=x.device, dtype=torch.int64) * self.stride
		x_offsets = torch.arange(out_w, device=x.device, dtype=torch.int64) * self.stride

		abs_h = ch[:, None, None, :] + y_offsets[None, :, None, None]
		abs_w = cw[:, None, None, :] + x_offsets[None, None, :, None]

		linear_idx = cc[:, None, None, :] * (height * width) + abs_h * width + abs_w
		x_flat = x.reshape(batch_size, -1)
		leaves = x_flat[:, linear_idx.reshape(-1)].reshape(batch_size, self.kernels, out_h, out_w, self.leaf_size)
		self._debug(f"forward random leaves_shape={tuple(leaves.shape)}")
		return leaves

	def _forward_learnable_connections(self, x, batch_size, out_h, out_w):
		cc_sel = self._st_select(self.Cc)
		ch_sel = self._st_select(self.Ch)
		cw_sel = self._st_select(self.Cw)
		self._debug(
			f"forward selected_mappings cc_sel={tuple(cc_sel.shape)}, ch_sel={tuple(ch_sel.shape)}, cw_sel={tuple(cw_sel.shape)}"
		)

		patches = torch.nn.functional.unfold(
			x,
			kernel_size=(self.receptive_field, self.receptive_field),
			stride=self.stride,
		)
		self._debug(f"forward unfold_output_shape={tuple(patches.shape)}")
		patches = patches.reshape(
			batch_size,
			self.in_channels,
			self.receptive_field,
			self.receptive_field,
			out_h * out_w,
		)
		self._debug(f"forward patches_reshaped_shape={tuple(patches.shape)}")

		leaves_per_kernel = []
		for kernel_idx in range(self.kernels):
			group_idx = int(self.kernel_groups[kernel_idx].item())
			c_start = group_idx * self.channels_per_group
			c_end = c_start + self.channels_per_group
			patch_k = patches[:, c_start:c_end, :, :, :]
			if self.debug and kernel_idx < 2:
				self._debug(
					f"forward kernel={kernel_idx} group={group_idx} patch_k_shape={tuple(patch_k.shape)}"
				)

			leaf_k = torch.einsum(
				"nchwl,ac,ah,aw->nal",
				patch_k,
				cc_sel[kernel_idx],
				ch_sel[kernel_idx],
				cw_sel[kernel_idx],
			)
			leaves_per_kernel.append(leaf_k.reshape(batch_size, self.leaf_size, out_h, out_w).permute(0, 2, 3, 1))

		leaves = torch.stack(leaves_per_kernel, dim=1)
		self._debug(f"forward stacked_leaves_shape={tuple(leaves.shape)}")
		return leaves

	def output_spatial_shape(self, height, width):
		effective_h = height + 2 * self.padding[0]
		effective_w = width + 2 * self.padding[1]
		if effective_h < self.receptive_field or effective_w < self.receptive_field:
			raise ValueError("Input spatial size after padding must be >= receptive_field.")

		out_h = 1 + (effective_h - self.receptive_field) // self.stride
		out_w = 1 + (effective_w - self.receptive_field) // self.stride
		return out_h, out_w

	def _evaluate_tree(self, leaves):
		# leaves: [N, K, Oh, Ow, leaf_size]
		self._debug(f"evaluate_tree input_leaves_shape={tuple(leaves.shape)}")
		n, k, oh, ow, leaf_size = leaves.shape
		state = leaves.permute(0, 2, 3, 1, 4).reshape(n * oh * ow, k * leaf_size)
		self._debug(f"evaluate_tree state_shape_after_reshape={tuple(state.shape)}")

		for level_idx, lut_layer in enumerate(self.lut_layers):
			self._debug(f"evaluate_tree level={level_idx} input_shape={tuple(state.shape)}")
			state = lut_layer(state)
			self._debug(f"evaluate_tree level={level_idx} output_shape={tuple(state.shape)}")

		out = state.reshape(n, oh, ow, k).permute(0, 3, 1, 2).contiguous()
		self._debug(f"evaluate_tree output_shape={tuple(out.shape)}")
		return out

	def forward(self, x):
		self._debug(f"forward start x_shape={tuple(x.shape)}, x_device={x.device}, x_dtype={x.dtype}")
		if x.ndim != 4:
			raise ValueError("Expected input shape [batch, channels, height, width].")
		if not x.is_cuda:
			raise ValueError("DWNConv currently requires CUDA because LUTLayer CPU path is not implemented.")

		batch_size, channels, height, width = x.shape
		if channels != self.in_channels:
			raise ValueError(f"Expected {self.in_channels} channels but got {channels}.")

		orig_h, orig_w = height, width
		pad_h, pad_w = self.padding
		if pad_h > 0 or pad_w > 0:
			x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
			self._debug(f"forward padded_x_shape={tuple(x.shape)}")

		batch_size, channels, height, width = x.shape
		out_h, out_w = self.output_spatial_shape(orig_h, orig_w)
		self._debug(f"forward output_spatial_shape out_h={out_h}, out_w={out_w}")

		if self.learnable_connections:
			leaves = self._forward_learnable_connections(x, batch_size, out_h, out_w)
		else:
			leaves = self._forward_random_connections(x, batch_size, out_h, out_w, height, width)

		roots = self._evaluate_tree(leaves)
		self._debug(f"forward roots_shape={tuple(roots.shape)}")

		if self.flatten_output:
			out = roots.reshape(batch_size, self.kernels * out_h * out_w)
			self._debug(f"forward output_flat_shape={tuple(out.shape)}")
			return out
		self._debug(f"forward output_shape={tuple(roots.shape)}")
		return roots

	def extra_repr(self):
		return (
			f"in_channels={self.in_channels}, depth={self.depth}, lut_rank={self.lut_rank}, "
			f"kernels={self.kernels}, receptive_field={self.receptive_field}, stride={self.stride}, "
			f"padding={self.padding}, channels_per_group={self.channels_per_group}, ste={self.ste}, flatten_output={self.flatten_output}, "
			f"random_kernel_groups={self.random_kernel_groups}, learnable_connections={self.learnable_connections}, "
			f"mapping_tau={self.mapping_tau}, debug={self.debug}"
		)


class LogicalORPool2d(torch.nn.Module):
	def __init__(self, kernel_size, stride):
		super().__init__()

		self.kernel_size = self._to_pair(kernel_size, "kernel_size")
		self.stride = self._to_pair(stride, "stride")

		if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
			raise ValueError("kernel_size values must be > 0.")
		if self.stride[0] <= 0 or self.stride[1] <= 0:
			raise ValueError("stride values must be > 0.")

	@staticmethod
	def _to_pair(value, name):
		if isinstance(value, int):
			return (value, value)
		if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
			return value
		raise ValueError(f"{name} must be an int or a tuple of 2 ints.")

	def output_spatial_shape(self, height, width):
		kh, kw = self.kernel_size
		sh, sw = self.stride
		if height < kh or width < kw:
			raise ValueError("Input spatial size must be >= kernel_size.")
		out_h = 1 + (height - kh) // sh
		out_w = 1 + (width - kw) // sw
		return out_h, out_w

	def forward(self, x):
		if x.ndim != 4:
			raise ValueError("Expected input shape [batch, channels, height, width].")

		_, _, height, width = x.shape
		self.output_spatial_shape(height, width)

		# Logical OR over each pooling window; for binary tensors this is max pooling.
		x_bool = (x > 0).to(dtype=x.dtype)
		return F.max_pool2d(x_bool, kernel_size=self.kernel_size, stride=self.stride)

	def extra_repr(self):
		return f"kernel_size={self.kernel_size}, stride={self.stride}"

