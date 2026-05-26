import argparse, os, math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchaudio

import torch_dwn as dwn

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate_last(x_bits, y, lut_layer, head_gp):
    x_map, map_idx = _mapping_forward_inputs(lut_layer, x_bits)
    raw = efd_cuda.forward(x_map.to(torch.float32), map_idx, lut_layer.luts.to(torch.float32))
    y_bits = (raw > 0).to(torch.float32)
    logits = head_gp.forward(y_bits)
    return evaluate_logits(logits, y)

def load_dataset(name):
    data_root = "./data"
    if name == "mnist":
        ds_train = datasets.MNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
        ds_test = datasets.MNIST(data_root, train=False, download=True, transform=transforms.ToTensor())
    elif name == "fashion":
        ds_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
        ds_test = datasets.FashionMNIST(data_root, train=False, download=True, transform=transforms.ToTensor())
    elif name == "kmnist":
        ds_train = datasets.KMNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
        ds_test = datasets.KMNIST(data_root, train=False, download=True, transform=transforms.ToTensor())
    elif name == "svhn":
        ds_train = datasets.SVHN(data_root, split="train", download=True, transform=transforms.ToTensor())
        ds_test = datasets.SVHN(data_root, split="test", download=True, transform=transforms.ToTensor())
    elif name == "cifar10":
        ds_train = datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.ToTensor())
        ds_test = datasets.CIFAR10(data_root, train=False, download=True, transform=transforms.ToTensor())
    elif name == "speechcommands":
        from torchaudio.datasets import SPEECHCOMMANDS

        core_keywords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

        class KWS10(SPEECHCOMMANDS):
            def __init__(self, root, subset):
                super().__init__(root=root, download=True, subset=subset)
                self.valid_indices = [i for i, path in enumerate(self._walker) if path.split(os.sep)[-2] in core_keywords]
                self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

            def __getitem__(self, index):
                waveform, sample_rate, label, *_ = super().__getitem__(self.valid_indices[index])
                feat = self.mel(waveform)
                feat = feat[..., :100]
                if feat.shape[-1] < 100:
                    feat = torch.nn.functional.pad(feat, (0, 100 - feat.shape[-1]))
                feat = feat / (feat.max() + 1e-6)
                return feat, core_keywords.index(label)

            def __len__(self):
                return len(self.valid_indices)

        ds_train = KWS10(root=data_root, subset="training")
        ds_test = KWS10(root=data_root, subset="testing")
    else:
        raise NotImplementedError(f"{name} not supported")

    train_loader = DataLoader(ds_train, batch_size=len(ds_train), shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
    return train_loader, test_loader

def infer_in_channels_and_size(sample_tensor):
    if sample_tensor.dim() == 4:
        sample_tensor = sample_tensor[0]
    C, H, W = sample_tensor.shape
    return int(C), int(H), int(W)

class DistributiveThermometer:
    def __init__(self, num_bits=1, feature_wise=False):
        self.num_bits = int(num_bits)
        self.feature_wise = bool(feature_wise)

    def get_thresholds(self, x):
        if not self.feature_wise:
            data = torch.sort(x.flatten())[0]
        else:
            data = torch.sort(x, dim=0)[0]

        idx = torch.tensor([int(data.shape[0] * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)])
        thr = data[idx]
        return torch.permute(thr, (*list(range(1, thr.ndim)), 0))

@torch.no_grad()
def thermo_encode_dataset(loader, device, bin_ch, feature_wise=False, is_speech=False):
    xb, _ = next(iter(loader))
    if feature_wise:
        if is_speech:
            x_for_thr = xb.squeeze(1)
            B, C, L = x_for_thr.shape
            x_2d = x_for_thr.permute(0, 2, 1).reshape(-1, C)
        else:
            B, C, H, W = xb.shape
            x_2d = xb.permute(0, 2, 3, 1).reshape(-1, C)

        thermo = DistributiveThermometer(num_bits=bin_ch, feature_wise=True)
        thresholds_all = thermo.get_thresholds(x_2d).to(device)
    else:
        flat = xb.flatten()
        thermo = DistributiveThermometer(num_bits=bin_ch, feature_wise=False)
        thresholds_all = thermo.get_thresholds(flat).to(device)

    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        if feature_wise:
            if is_speech:
                x_s = x.squeeze(1).unsqueeze(2)
                thr = thresholds_all.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                xb_bin = (x_s.unsqueeze(2) > thr).to(torch.float32)
            else:
                Bc, Cc, Hc, Wc = x.shape
                thr = thresholds_all.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                xb_bin = (x.unsqueeze(2) > thr).to(torch.float32)
        else:
            thr = thresholds_all.view(bin_ch, 1, 1).to(device)
            xb_bin = (x.unsqueeze(2) > thr.view(1, 1, bin_ch, 1, 1)).to(torch.float32)

        z = xb_bin.flatten(1)
        feats.append(z.cpu())
        labels.append(y.clone())
    X = torch.cat(feats, dim=0).contiguous()
    Y = torch.cat(labels, dim=0).contiguous()
    return X, Y

def _make_branch_patterns(S: int, T: int, pattern_seed: int = 1234):
    import torch

    N = S * S
    g = torch.Generator().manual_seed(int(pattern_seed))
    all_combs = torch.combinations(torch.arange(N), r=6)
    T = min(T, all_combs.size(0))
    idx = torch.randperm(all_combs.size(0), generator=g)[:T]
    return all_combs[idx].sort(dim=1).values

def _make_branch_patterns_1d(S: int, T: int, pattern_seed: int = 1234):
    import torch

    assert S >= 6, "1×S"
    g = torch.Generator().manual_seed(int(pattern_seed))
    all_combs = torch.combinations(torch.arange(S), r=6)
    T = min(T, all_combs.size(0))
    idx = torch.randperm(all_combs.size(0), generator=g)[:T]
    return all_combs[idx].sort(dim=1).values

class RGB_SxS_Conv6LUT_Encoder(nn.Module):
    def __init__(self, bin_channels: int, in_channels: int = 3, S: int = 5, T: int = 4, thresholds=None, pattern_seed: int = 1234, stride: int = 1, padding: int = 0):
        super().__init__()
        self.Bthr = int(bin_channels)
        self.C = int(in_channels)
        self.S = int(S)
        self.T = int(T)
        self.ks = (self.S, self.S)
        self.stride = stride
        self.padding = padding

        self.tables = nn.Parameter(torch.zeros(self.Bthr, self.C, self.T, 64), requires_grad=False)
        nn.init.uniform_(self.tables, -0.05, 0.05)

        if thresholds is None:
            thresholds = [(k + 1) / float(self.Bthr + 1) for k in range(self.Bthr)]
            thresholds = torch.tensor(thresholds)
        self.thresholds = thresholds

        patterns = _make_branch_patterns(self.S, self.T, pattern_seed=pattern_seed)
        self.register_buffer("patterns", patterns)
        self.register_buffer("bit_weights", torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.float32).view(1, 1, 6))
        self._last_indices = None
        self._last_grad_y = None

    def _binarize(self, x):
        t = self.thresholds.to(dtype=x.dtype, device=x.device)
        if t.ndim == 1:
            t_view = t.view(1, 1, self.Bthr, 1, 1)
        elif t.ndim == 2:
            assert t.shape[0] == self.C and t.shape[1] == self.Bthr, f"thresholds shape {t.shape} and (C={self.C},B={self.Bthr}) not support"
            t_view = t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"not support: {t.shape}, [C,Bthr]")

        return (x.unsqueeze(2) > t_view).to(x.dtype)

    def _unfold_and_index(self, xb):
        Bsz, C, Bthr, H, W = xb.shape
        patterns = self.patterns
        idx_list = []
        for t in range(Bthr):
            for c in range(C):
                x_tc = xb[:, c : c + 1, t, :, :]
                patches = F.unfold(x_tc, kernel_size=self.ks, padding=self.padding, stride=self.stride)
                patches = patches.view(Bsz, self.S * self.S, -1)
                for b in range(self.T):
                    sel = patterns[b].to(patches.device)
                    bits = patches[:, sel, :]
                    idx = (bits.transpose(1, 2) * self.bit_weights.to(patches.dtype)).sum(dim=2).to(torch.int64)
                    idx_list.append(idx)
        return idx_list

    def forward(self, x):
        xb = self._binarize(x)
        idx_list = self._unfold_and_index(xb)
        Bsz, L = idx_list[0].shape
        H, W = x.shape[-2:]
        Hout = (H - self.ks[0]) // self.stride + 1
        Wout = (W - self.ks[1]) // self.stride + 1

        y_sum = torch.zeros(Bsz, self.Bthr * self.C * self.T, L, device=x.device, dtype=x.dtype)
        flat = 0
        for t in range(self.Bthr):
            for c in range(self.C):
                for b in range(self.T):
                    idx = idx_list[flat]
                    lut = self.tables[t, c, b, :].clamp(-1.0, 1.0)
                    y_sum[:, flat, :] = lut[idx]
                    flat += 1

        y_cont = y_sum.view(Bsz, self.Bthr * self.C * self.T, Hout, Wout)
        y_bin_raw = (y_cont > 0).to(x.dtype)

        if self.training:
            y_bin = y_bin_raw.detach().requires_grad_(True)
            self._last_indices = [idx.detach() for idx in idx_list]

            def _save_grad(g):
                self._last_grad_y = g.detach().view(Bsz, self.Bthr * self.C * self.T, -1)

            y_bin.register_hook(_save_grad)
            return y_bin
        else:
            self._last_indices = None
            self._last_grad_y = None
            return y_bin_raw

    @torch.no_grad()
    def update_tables_from_grad(self, lr: float = 1.0, ema: float = 0.0):
        if (self._last_indices is None) or (self._last_grad_y is None):
            return
        idx_list = self._last_indices
        grad = self._last_grad_y
        flat = 0
        for t in range(self.Bthr):
            for c in range(self.C):
                for b in range(self.T):
                    e_flat = idx_list[flat].reshape(-1)
                    w = grad[:, flat, :].reshape(-1)
                    G = torch.bincount(e_flat, weights=w, minlength=64).to(self.tables.dtype)
                    if ema > 0.0:
                        self.tables[t, c, b, :].mul_(ema).add_(-lr * G, alpha=(1.0 - ema))
                    else:
                        self.tables[t, c, b, :].add_(-lr * G)
                    flat += 1
        self.tables.clamp_(-1.0, 1.0)
        self._last_indices = None
        self._last_grad_y = None

class Speech_1xS_Conv6LUT_Encoder(nn.Module):
    def __init__(self, bin_channels: int, in_channels: int = 64, S: int = 5, T: int = 4, thresholds=None, pattern_seed: int = 1234, stride: int = 1, padding: int = 0):
        super().__init__()
        self.Bthr = int(bin_channels)
        self.C = int(in_channels)
        self.S = int(S)
        self.T = int(T)
        self.ks = (1, self.S)
        self.stride = stride
        self.padding = padding

        self.tables = nn.Parameter(torch.zeros(self.Bthr, self.C, self.T, 64), requires_grad=False)
        nn.init.uniform_(self.tables, -0.05, 0.05)

        if thresholds is None:
            thresholds = [(k + 1) / float(self.Bthr + 1) for k in range(self.Bthr)]
            thresholds = torch.tensor(thresholds)
        self.thresholds = thresholds

        patterns = _make_branch_patterns_1d(self.S, self.T, pattern_seed=pattern_seed)
        self.register_buffer("patterns", patterns)
        self.register_buffer("bit_weights", torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.float32).view(1, 1, 6))
        self._last_indices = None
        self._last_grad_y = None

    def _binarize(self, x):
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 64:
            x = x.squeeze(1)
            x = x.unsqueeze(2)
        elif x.dim() == 4 and x.shape[2] == 1:
            pass
        else:
            raise ValueError(f"Speech_1xS_Conv6LUT_Encoder input [B,1,64,W] or [B,64,1,W]，get {x.shape}")

        t = self.thresholds.to(dtype=x.dtype, device=x.device)
        if t.ndim == 1:
            t_view = t.view(1, 1, self.Bthr, 1, 1)
        elif t.ndim == 2:
            assert t.shape[0] == self.C and t.shape[1] == self.Bthr, f"thresholds shape {t.shape} and (C={self.C},B={self.Bthr}) not match"
            t_view = t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"not support thresholds : {t.shape},  [Bthr] or [C,Bthr]")

        xb = (x.unsqueeze(2) > t_view).to(x.dtype)
        return xb

    def _unfold_and_index(self, xb):
        Bsz, C, Bthr, H, W = xb.shape
        patterns = self.patterns
        idx_list = []
        for t in range(Bthr):
            for c in range(C):
                x_tc = xb[:, c : c + 1, t, :, :]
                patches = F.unfold(x_tc, kernel_size=self.ks, padding=self.padding, stride=self.stride)
                patches = patches.view(Bsz, self.S, -1)
                for b in range(self.T):
                    sel = patterns[b].to(patches.device)
                    bits = patches[:, sel, :]
                    idx = (bits.transpose(1, 2) * self.bit_weights.to(patches.dtype)).sum(dim=2).to(torch.int64)
                    idx_list.append(idx)
        return idx_list

    def forward(self, x):
        xb = self._binarize(x)
        idx_list = self._unfold_and_index(xb)
        Bsz, L = idx_list[0].shape
        W = xb.shape[-1]
        Hout = 1
        Wout = (W - self.S) // self.stride + 1

        y_sum = torch.zeros(Bsz, self.Bthr * self.C * self.T, L, device=xb.device, dtype=xb.dtype)
        flat = 0
        for t in range(self.Bthr):
            for c in range(self.C):
                for b in range(self.T):
                    idx = idx_list[flat]
                    lut = self.tables[t, c, b, :].clamp(-1.0, 1.0)
                    y_sum[:, flat, :] = lut[idx]
                    flat += 1

        y_cont = y_sum.view(Bsz, self.Bthr * self.C * self.T, Hout, Wout)
        y_bin_raw = (y_cont > 0).to(x.dtype)

        if self.training:
            y_bin = y_bin_raw.detach().requires_grad_(True)
            self._last_indices = [idx.detach() for idx in idx_list]

            def _save_grad(g):
                self._last_grad_y = g.detach().view(Bsz, self.Bthr * self.C * self.T, -1)

            y_bin.register_hook(_save_grad)
            return y_bin
        else:
            self._last_indices = None
            self._last_grad_y = None
            return y_bin_raw

    @torch.no_grad()
    def update_tables_from_grad(self, lr: float = 1.0, ema: float = 0.0):
        if (self._last_indices is None) or (self._last_grad_y is None):
            return
        idx_list = self._last_indices
        grad = self._last_grad_y
        flat = 0
        for t in range(self.Bthr):
            for c in range(self.C):
                for b in range(self.T):
                    e_flat = idx_list[flat].reshape(-1)
                    w = grad[:, flat, :].reshape(-1)
                    G = torch.bincount(e_flat, weights=w, minlength=64).to(self.tables.dtype)
                    if ema > 0.0:
                        self.tables[t, c, b, :].mul_(ema).add_(-lr * G, alpha=(1.0 - ema))
                    else:
                        self.tables[t, c, b, :].add_(-lr * G)
                    flat += 1
        self.tables.clamp_(-1.0, 1.0)
        self._last_indices = None
        self._last_grad_y = None

class _BinaryOrPoolFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size: int, threshold_q: int):
        k = int(kernel_size)
        q = int(threshold_q)
        B, C, H, W = x.shape
        Hout = (H - k) // k + 1
        Wout = (W - k) // k + 1
        patches = F.unfold(x, kernel_size=k, stride=k)
        L = patches.shape[-1]
        patches = patches.view(B, C, k * k, L)
        cnt1 = (patches > 0).sum(dim=2)
        y = (cnt1 > q).to(x.dtype).view(B, C, Hout, Wout)
        ctx.save_for_backward(x)
        ctx.k = k
        ctx.q = q
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        k = ctx.k
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=k, stride=k)
        patches = patches.view(B, C, k * k, -1)
        ones_mask = patches > 0
        e = ones_mask.sum(dim=2, keepdim=True)

        grad_out = grad_out.contiguous()
        Hout = (H - k) // k + 1
        Wout = (W - k) // k + 1
        go = grad_out.view(B, C, Hout * Wout)

        scale = torch.zeros_like(e, dtype=patches.dtype)
        mask_pos = e > 0
        scale[mask_pos] = (go.unsqueeze(2) / e)[mask_pos]

        gin_patches = torch.where(ones_mask, scale, torch.zeros_like(scale))
        gin = gin_patches.view(B, C * k * k, -1)
        grad_in = F.fold(gin, output_size=(H, W), kernel_size=k, stride=k)
        return grad_in, None, None

class BinaryOrPool(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold_q: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.threshold_q = int(threshold_q)

    def forward(self, x):
        return _BinaryOrPoolFn.apply(x, self.kernel_size, self.threshold_q)

class _BinaryOrPool1DFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size: int, threshold_q: int):
        k = int(kernel_size)
        q = int(threshold_q)
        B, C, H, W = x.shape
        assert H == 1, f"BinaryOrPool1D H=1, get H={H}"
        Hout = 1
        Wout = (W - k) // k + 1
        patches = F.unfold(x, kernel_size=(1, k), stride=(1, k))
        L = patches.shape[-1]
        patches = patches.view(B, C, k, L)
        cnt1 = (patches > 0).sum(dim=2)
        y = (cnt1 > q).to(x.dtype).view(B, C, Hout, Wout)
        ctx.save_for_backward(x)
        ctx.k = k
        ctx.q = q
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        k = ctx.k
        B, C, H, W = x.shape
        assert H == 1
        patches = F.unfold(x, kernel_size=(1, k), stride=(1, k))
        patches = patches.view(B, C, k, -1)
        ones_mask = patches > 0
        e = ones_mask.sum(dim=2, keepdim=True)

        grad_out = grad_out.contiguous()
        Hout = 1
        Wout = (W - k) // k + 1
        go = grad_out.view(B, C, Hout * Wout)

        scale = torch.zeros_like(e, dtype=patches.dtype)
        mask_pos = e > 0
        scale[mask_pos] = (go.unsqueeze(2) / e)[mask_pos]

        gin_patches = torch.where(ones_mask, scale, torch.zeros_like(scale))
        gin = gin_patches.view(B, C * k, -1)
        grad_in = F.fold(gin, output_size=(H, W), kernel_size=(1, k), stride=(1, k))
        return grad_in, None, None

class BinaryOrPool1D(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold_q: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.threshold_q = int(threshold_q)

    def forward(self, x):
        return _BinaryOrPool1DFn.apply(x, self.kernel_size, self.threshold_q)

class FactoredLinear(nn.Module):
    def __init__(self, D, H, r=1536):
        super().__init__()
        self.A = nn.Linear(D, r, bias=False)
        self.B = nn.Linear(r, H, bias=True)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.bias)

    def forward(self, x):
        return self.B(self.A(x))

class EncoderMLPTeacher(nn.Module):
    def __init__(self, bin_channels=16, in_channels=3, S=5, T=4, mlp_hidden=1024, pattern_seed=1234, pool_ks: int = 2, pool_q: int = 0, mlp_rank: int = 1536, input_h: int = 32, input_w: int = 32):
        super().__init__()
        self.Bthr = bin_channels
        self.S = S
        self.T = T
        self.C = in_channels

        self.enc = RGB_SxS_Conv6LUT_Encoder(
            bin_channels=self.Bthr,
            in_channels=self.C,
            S=self.S,
            T=self.T,
            thresholds=[(k + 1) / float(self.Bthr + 1) for k in range(self.Bthr)],
            pattern_seed=pattern_seed,
            stride=1,
            padding=0,
        )

        self.in_h, self.in_w = int(input_h), int(input_w)
        self.out_h, self.out_w = (self.in_h - self.S + 1, self.in_w - self.S + 1)

        self.pool = BinaryOrPool(kernel_size=pool_ks, threshold_q=pool_q)
        self.pool_k = int(pool_ks)
        self.pool_q = int(pool_q)
        self.pool_h = (self.out_h - self.pool_k) // self.pool_k + 1
        self.pool_w = (self.out_w - self.pool_k) // self.pool_k + 1

        feat_dim = self.Bthr * self.C * self.T * self.pool_h * self.pool_w
        self._feat_dim = feat_dim

        self.mlp_first = FactoredLinear(feat_dim, mlp_hidden, r=mlp_rank)
        self.mlp_rest = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(mlp_hidden, 10))
        for m in self.mlp_rest:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

        self._printed_parallel = False

    def forward(self, x):
        yb = self.enc(x)
        yp = self.pool(yb)
        z = yp.flatten(1)
        if self.training and (not self._printed_parallel):
            print(f"[Encoder bits after pool] width = {z.shape[1]} " f"(= B({self.Bthr})*C({self.C})*T({self.T})*{self.pool_h}*{self.pool_w}, Q={self.pool_q})")
            self._printed_parallel = True
        h = self.mlp_first(z)
        logits = self.mlp_rest(h)
        return logits

class EncoderMLPTeacherSpeech(nn.Module):
    def __init__(self, bin_channels=16, S=5, T=4, mlp_hidden=1024, pattern_seed=1234, pool_ks: int = 2, pool_q: int = 0, mlp_rank: int = 1536, input_len: int = 100, in_channels: int = 64):
        super().__init__()
        self.Bthr = bin_channels
        self.S = S
        self.T = T
        self.C = in_channels

        self.enc = Speech_1xS_Conv6LUT_Encoder(
            bin_channels=self.Bthr,
            in_channels=self.C,
            S=self.S,
            T=self.T,
            thresholds=[(k + 1) / float(self.Bthr + 1) for k in range(self.Bthr)],
            pattern_seed=pattern_seed,
            stride=1,
            padding=0,
        )

        self.in_len = int(input_len)
        self.out_len = self.in_len - self.S + 1

        self.pool = BinaryOrPool1D(kernel_size=pool_ks, threshold_q=pool_q)
        self.pool_k = int(pool_ks)
        self.pool_q = int(pool_q)
        self.pool_h = 1
        self.pool_w = (self.out_len - self.pool_k) // self.pool_k + 1

        feat_dim = self.Bthr * self.C * self.T * self.pool_h * self.pool_w
        self._feat_dim = feat_dim

        self.mlp_first = FactoredLinear(feat_dim, mlp_hidden, r=mlp_rank)
        self.mlp_rest = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(mlp_hidden, 10))
        for m in self.mlp_rest:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

        self._printed_parallel = False

    def forward(self, x):
        yb = self.enc(x)
        yp = self.pool(yb)
        z = yp.flatten(1)
        if self.training and (not self._printed_parallel):
            print(f"[Encoder(Speech) bits after pool] width = {z.shape[1]} " f"(= B({self.Bthr})*C({self.C})*T({self.T})*{self.pool_h}*{self.pool_w}, Q={self.pool_q})")
            self._printed_parallel = True
        h = self.mlp_first(z)
        logits = self.mlp_rest(h)
        return logits

def get_generic_loaders_p1(ds_train, ds_test, batch_size=128, num_workers=2, augment=False, dataset_name="cifar10"):
    if dataset_name == "cifar10" and augment:
        tf_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose(tf_train))
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    else:
        train_ds = ds_train
        test_ds = ds_test

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def train_phase1_teacher(model, train_loader, test_loader, device, epochs=150, mlp_lr=1e-4, table_lr=1e-3, table_ema=0.0):
    mlp_params = list(model.mlp_first.parameters()) + list(model.mlp_rest.parameters())
    opt = torch.optim.Adam(mlp_params, lr=mlp_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=mlp_lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    amp_dtype = torch.bfloat16 if (device.type == "cuda") else None

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            opt.step()

            model.enc.update_tables_from_grad(lr=table_lr, ema=table_ema)

        model.eval()
        acc = 0.0
        n = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc += (logits.argmax(1) == y).sum().item()
            n += y.numel()
        acc = acc / n
        best = max(best, acc)
        if (ep % 5 == 0) or (ep == 1):
            print(f"[P1] Epoch {ep:03d} | TestAcc={acc*100:.2f}% | Best={best*100:.2f}%")
        sched.step()

    return best

@torch.no_grad()
def encode_with_trained_encoder(encoder, pool, loader, device, Bthr, T, S, pool_k):
    encoder.eval()
    pool.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        yb = encoder(x)
        yp = pool(yb)
        z = yp.flatten(1).float().cpu()
        feats.append(z)
        labels.append(y.clone())
    X = torch.cat(feats, dim=0).contiguous()
    Y = torch.cat(labels, dim=0).contiguous()
    return X, Y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from torch_dwn import efd_cuda
except ModuleNotFoundError:
    import efd_cuda

for fn in ("forward", "backward"):
    assert hasattr(efd_cuda, fn), f"efd_cuda not {fn}"

@torch.no_grad()
def _is_learnable_mapping(lut_layer):
    return isinstance(getattr(lut_layer, "mapping", None), dwn.mapping.LearnableMapping)

@torch.no_grad()
def _mapping_forward_inputs(lut_layer, x_in):
    if _is_learnable_mapping(lut_layer):
        x_map = lut_layer.mapping(x_in)
        dummy = getattr(lut_layer, "__dummy_mapping", None)
        if dummy is None:
            dummy = getattr(lut_layer, "_LUTLayer__dummy_mapping", None)
        if dummy is None:
            dummy = getattr(lut_layer, "_LUTLayer__LUTLayer__dummy_mapping", None)
        assert dummy is not None, "LUTLayer(learnable) not find dummy mapping"
        mapping_idx = dummy.to(torch.int32).contiguous().to(device)
        return x_map, mapping_idx
    else:
        mapping_idx = lut_layer.mapping.to(torch.int32).contiguous().to(device)
        return x_in, mapping_idx

@torch.no_grad()
def _mapping_weights_grad(lut_layer, x_in, dy_after_map):
    xpm = 2 * x_in - 1.0
    grad_w = xpm.t().matmul(dy_after_map)
    return grad_w.contiguous()

class LearnableGP(nn.Module):
    def __init__(self, in_features, num_classes, connect_ratio, tau):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.connect_ratio = float(connect_ratio)
        self.tau = float(tau)

        self.mapping_logits = nn.Parameter(torch.empty(num_classes, in_features, device=device, dtype=torch.float32))
        nn.init.uniform_(self.mapping_logits, -0.1, 0.1)

        self._last_indices = None
        self._last_input_bits = None

    @torch.no_grad()
    def _topk_indices(self):
        K = max(1, int(self.in_features * self.connect_ratio))
        return torch.topk(self.mapping_logits, K, dim=1, largest=True, sorted=False).indices

    def forward(self, x_bits):
        self._last_input_bits = x_bits.detach()
        idx = self._topk_indices()
        self._last_indices = [idx[c].detach() for c in range(self.num_classes)]
        outs = []
        for c in range(self.num_classes):
            sel = self._last_indices[c]
            y_sel = x_bits.index_select(1, sel)
            outs.append(y_sel.sum(dim=1) / self.tau)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def backward_update(self, grad_cls, lr):
        assert self._last_input_bits is not None, "call forward() before backward_update()"
        S = self._last_input_bits.mul(2.0).add_(-1.0).to(torch.float32)
        GO = grad_cls.to(torch.float32)
        mapping_grad = torch.matmul(GO.t(), S)
        self.mapping_logits.add_(-lr * mapping_grad)

    @torch.no_grad()
    def last_indices(self):
        assert self._last_indices is not None, "need forward() first"
        return self._last_indices

def train_single_layer(x_train, y_train, x_test, y_test, lut_width, n_tuple, tau_groupsum, lr_last_layer, lr_gp, connect_ratio=0.1, mapping_type="learnable", clamp_luts=True, batch_size=128, epochs=100, eval_every=5):
    lut = dwn.LUTLayer(x_train.size(1), lut_width, n=n_tuple, mapping=mapping_type, clamp_luts=clamp_luts, ste=False).to(device)
    lut.luts.requires_grad_(False)
    if _is_learnable_mapping(lut):
        lut.mapping.weights.requires_grad_(False)

    gp = LearnableGP(in_features=lut_width, num_classes=10, connect_ratio=connect_ratio, tau=tau_groupsum).to(device)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(x_train.size(0), device=device)
        loss_sum, top1_sum, steps = 0.0, 0.0, 0

        for i0 in range(0, x_train.size(0), batch_size):
            idx = perm[i0 : i0 + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)

            x_map, map_idx = _mapping_forward_inputs(lut, xb)
            y_raw = efd_cuda.forward(x_map, map_idx, lut.luts)
            y_bits = (y_raw > 0).to(torch.float32)

            logits = gp.forward(y_bits)
            loss = F.cross_entropy(logits, yb)

            p = torch.softmax(logits, dim=1)
            one_hot = torch.zeros_like(p).scatter_(1, yb.view(-1, 1), 1.0)
            g_cls = p - one_hot

            dY = torch.zeros_like(y_bits)
            idx_sel = gp.last_indices()
            for c in range(10):
                sel = idx_sel[c]
                dY[:, sel] += g_cls[:, c].unsqueeze(1) / tau_groupsum

            in_grad, luts_grad = efd_cuda.backward(x_map, map_idx, lut.luts, lut.alpha, lut.beta, dY)
            lut.luts.add_(-lr_last_layer * luts_grad)
            if clamp_luts:
                lut.luts.clamp_(-1, 1)

            if _is_learnable_mapping(lut):
                grad_w = _mapping_weights_grad(lut, xb, in_grad)
                lut.mapping.weights.add_(-lr_last_layer * grad_w)

            gp.backward_update(g_cls, lr=lr_gp)

            loss_sum += loss.item()
            top1_sum += ((logits.argmax(1) == yb).float().mean().item())
            steps += 1

        if (ep % eval_every == 0) or (ep == 1):
            acc = evaluate_last(x_test, y_test, lut, gp)
            print(f"[Single-LUT] Ep {ep:03d}/{epochs} | " f"CE {loss_sum/steps:.4f} | BatchTop1 {top1_sum/steps:.4f} | TestAcc {acc:.4f}")

    final_acc = evaluate_last(x_test, y_test, lut, gp)
    print(f"[Final Accuracy] {final_acc:.4f}")
    return final_acc
    #return final_acc, lut, gp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10", help="mnist/fashion/kmnist/svhn/cifar10/speechcommands")

    ap.add_argument("--use_augment", action="store_true")
    ap.add_argument("--p1_epochs", type=int, default=500)
    ap.add_argument("--p1_batch_size", type=int, default=128)
    ap.add_argument("--p1_mlp_lr", type=float, default=1e-4)
    ap.add_argument("--p1_table_lr", type=float, default=1e-3)
    ap.add_argument("--p1_table_ema", type=float, default=0.0)

    ap.add_argument("--bin_ch", type=int, default=4)
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--T", type=int, default=36)
    ap.add_argument("--pattern_seed", type=int, default=1234)
    ap.add_argument("--mlp_hidden", type=int, default=12000)
    ap.add_argument("--pool_ks", type=int, default=5)
    ap.add_argument("--Q", type=int, default=5)
    ap.add_argument("--p1_mlp_rank", type=int, default=8000)

    ap.add_argument("--thr_mode", type=str, default="per_channel", choices=["global", "per_channel"], help="global or per_channel")

    ap.add_argument("--fd_batch_size", type=int, default=128)
    ap.add_argument("--width", type=int, default=8000, help="width of WNN")
    ap.add_argument("--K", type=int, default=6, help="LUT n")
    ap.add_argument("--tau_groupsum", type=float, default=50.0)
    ap.add_argument("--lr_last_layer", type=float, default=5e-3)
    ap.add_argument("--lr_gp", type=float, default=5e-3)
    ap.add_argument("--connect_ratio", type=float, default=0.2, help="LearnableGP ratio")
    ap.add_argument("--lut_mapping", type=str, default="learnable", choices=["learnable", "random", "arange"])
    ap.add_argument("--clamp_luts", action="store_true", default=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    set_seed(42)
    dev = torch.device(args.device)

    tr_full_loader, te_full_loader = load_dataset(args.dataset)
    ds_train = tr_full_loader.dataset
    ds_test = te_full_loader.dataset

    sample_x, _ = next(iter(tr_full_loader))
    in_channels, in_h, in_w = infer_in_channels_and_size(sample_x)

    train_loader_p1, test_loader_p1 = get_generic_loaders_p1(ds_train, ds_test, batch_size=args.p1_batch_size, num_workers=2, augment=args.use_augment, dataset_name=args.dataset)

    if args.dataset == "speechcommands":
        model_p1 = EncoderMLPTeacherSpeech(
            bin_channels=args.bin_ch,
            S=args.S,
            T=args.T,
            mlp_hidden=args.mlp_hidden,
            pattern_seed=args.pattern_seed,
            pool_ks=args.pool_ks,
            pool_q=args.Q,
            mlp_rank=args.p1_mlp_rank,
            input_len=in_w,
            in_channels=64,
        ).to(dev)
    else:
        model_p1 = EncoderMLPTeacher(
            bin_channels=args.bin_ch,
            in_channels=in_channels,
            S=args.S,
            T=args.T,
            mlp_hidden=args.mlp_hidden,
            pattern_seed=args.pattern_seed,
            pool_ks=args.pool_ks,
            pool_q=args.Q,
            mlp_rank=args.p1_mlp_rank,
            input_h=in_h,
            input_w=in_w,
        ).to(dev)

    feature_wise = args.thr_mode == "per_channel"
    with torch.no_grad():
        xb, _ = next(iter(train_loader_p1))
        xb = xb.to(dev)
        if feature_wise:
            if args.dataset == "speechcommands":
                x_for_thr = xb.squeeze(1)
                B, C, L = x_for_thr.shape
                x_2d = x_for_thr.permute(0, 2, 1).reshape(-1, C)
            else:
                B, C, H, W = xb.shape
                x_2d = xb.permute(0, 2, 3, 1).reshape(-1, C)

            thermo = DistributiveThermometer(num_bits=args.bin_ch, feature_wise=True)
            thresholds_all = thermo.get_thresholds(x_2d).to(dev)
        else:
            flat = xb.flatten()
            thermo = DistributiveThermometer(num_bits=args.bin_ch, feature_wise=False)
            thresholds_all = thermo.get_thresholds(flat).to(dev)

        model_p1.enc.thresholds = thresholds_all

    print(f"\n== Phase-1: Train Conv-LUT6+Pool+MLP on {args.dataset} | thr_mode={args.thr_mode} ==")
    best_p1 = train_phase1_teacher(model_p1, train_loader_p1, test_loader_p1, dev, epochs=args.p1_epochs, mlp_lr=args.p1_mlp_lr, table_lr=args.p1_table_lr, table_ema=args.p1_table_ema)
    print(f"[Phase-1 Best] TestAcc: {best_p1*100:.2f}%")

    encoder = model_p1.enc
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    pool = model_p1.pool
    pool.eval()

    enc_train_loader = DataLoader(ds_train, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    enc_test_loader = DataLoader(ds_test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    x_train_cpu, y_train_cpu = encode_with_trained_encoder(encoder, pool, enc_train_loader, dev, args.bin_ch, args.T, args.S, args.pool_ks)
    x_test_cpu, y_test_cpu = encode_with_trained_encoder(encoder, pool, enc_test_loader, dev, args.bin_ch, args.T, args.S, args.pool_ks)
    print(f"[Encoded] Train: {x_train_cpu.shape} | Test: {x_test_cpu.shape}")

    x_train = x_train_cpu.to(dev).contiguous().float()
    y_train = y_train_cpu.to(dev)
    x_test = x_test_cpu.to(dev).contiguous().float()
    y_test = y_test_cpu.to(dev)

    print("\n== Phase-2 (Single WNN + LearnableGP) ==")
#    _, lut, gp = train_single_layer(
    _ = train_single_layer(
        x_train,
        y_train,
        x_test,
        y_test,
        lut_width=args.width,
        n_tuple=args.K,
        tau_groupsum=args.tau_groupsum,
        lr_last_layer=args.lr_last_layer,
        lr_gp=args.lr_gp,
        connect_ratio=args.connect_ratio,
        mapping_type=args.lut_mapping,
        clamp_luts=args.clamp_luts,
        batch_size=args.fd_batch_size,
        epochs=5000,
        eval_every=5,
    )
    #with torch.no_grad():
    #    entry = lut.luts.detach().cpu()
    #    torch.save(entry, "Entry.pt")
    #    if _is_learnable_mapping(lut):
    #        lut_wire = lut.mapping.weights.detach().cpu()
    #    else:
    #        lut_wire = lut.mapping.detach().cpu()
    #    gp_idx = gp._topk_indices().detach().cpu()
    #    wire = {
    #        "lut_mapping": lut_wire,
    #        "gp_indices": gp_idx,
    #    }
    #    torch.save(wire, "Wire.pt")

if __name__ == "__main__":
    main()



