import torch

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True, channel_wise=False):
        
        assert num_bits > 0
        assert type(feature_wise) is bool
        assert type(channel_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.channel_wise = channel_wise
        self.thresholds = None

    def _threshold_steps(self, x):
        dtype = x.dtype if x.is_floating_point() else torch.float32
        return torch.arange(1, self.num_bits + 1, device=x.device, dtype=dtype)

    def get_thresholds(self, x):
        steps = self._threshold_steps(x)

        if self.channel_wise:
            if x.ndim != 4:
                raise ValueError('channel_wise thresholding expects input shape [batch, channels, height, width]')
            min_value = x.amin(dim=(0, 2, 3))
            max_value = x.amax(dim=(0, 2, 3))
            return min_value.unsqueeze(-1) + steps.unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + steps.unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self
    
    def binarize(self, x):
        if self.thresholds is None:
            raise 'need to fit before calling apply'
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        if self.channel_wise:
            if x.ndim != 4:
                raise ValueError('channel_wise thresholding expects input shape [batch, channels, height, width]')
            thresholds = self.thresholds.to(device=x.device, dtype=x.dtype if x.is_floating_point() else torch.float32)
            bits = (x.unsqueeze(2) > thresholds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).float()
            batch_size, channels, num_bits, height, width = bits.shape
            return bits.reshape(batch_size, channels * num_bits, height, width)

        x = x.unsqueeze(-1)
        return (x > self.thresholds.to(device=x.device)).float()

class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True, channel_wise=False):
        super().__init__(num_bits, feature_wise, channel_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(self._threshold_steps(x) / (self.num_bits + 1))

        if self.channel_wise:
            if x.ndim != 4:
                raise ValueError('channel_wise thresholding expects input shape [batch, channels, height, width]')
            mean = x.mean(dim=(0, 2, 3))
            std = x.std(dim=(0, 2, 3))
            return torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)

        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std() 
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds
    
class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True, channel_wise=False):
        super().__init__(num_bits, feature_wise, channel_wise)

    def get_thresholds(self, x):
        if self.channel_wise:
            if x.ndim != 4:
                raise ValueError('channel_wise thresholding expects input shape [batch, channels, height, width]')
            data = torch.sort(x.permute(1, 0, 2, 3).reshape(x.shape[1], -1), dim=1)[0]
            indicies = torch.tensor(
                [int(data.shape[1] * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)],
                device=x.device,
            )
            return data[:, indicies]

        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(self.num_bits+1)) for i in range(1, self.num_bits+1)], device=x.device)
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

