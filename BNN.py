import argparse
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


class SignSTE(torch.autograd.Function):
	"""Binary sign with straight-through gradient estimator."""

	@staticmethod
	def forward(ctx, x: torch.Tensor) -> torch.Tensor:
		ctx.save_for_backward(x)
		return x.sign().masked_fill(x == 0, 1.0)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
		(x,) = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[x.abs() > 1.0] = 0.0
		return grad_input


def binarize(x: torch.Tensor) -> torch.Tensor:
	return SignSTE.apply(x)


class BinaryActivation(nn.Module):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return binarize(x)


class BinaryLinear(nn.Linear):
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		weight = binarize(self.weight)
		return F.linear(input, weight, self.bias)


class BinaryConv2d(nn.Conv2d):
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		weight = binarize(self.weight)
		return F.conv2d(
			input,
			weight,
			self.bias,
			self.stride,
			self.padding,
			self.dilation,
			self.groups,
		)


class MLP(nn.Module):
	def __init__(self, input_dim: int, num_classes: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Flatten(),
			BinaryLinear(input_dim, 1024, bias=False),
			nn.BatchNorm1d(1024),
			BinaryActivation(),
			nn.Dropout(0.2),
			BinaryLinear(1024, 1024, bias=False),
			nn.BatchNorm1d(1024),
			BinaryActivation(),
			nn.Dropout(0.2),
			BinaryLinear(1024, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@torch.no_grad()
def clip_weights(model: nn.Module, min_val: float = -1.0, max_val: float = 1.0) -> None:
	for module in model.modules():
		if isinstance(module, BinaryLinear):
			module.weight.clamp_(min_val, max_val)


def get_dataloaders(dataset_name: str, data_dir: str, batch_size: int):
	dataset_name = dataset_name.lower()

	if dataset_name == "mnist":
		mean, std, channels, classes = (0.1307,), (0.3081,), 1, 10
		ds_class = datasets.MNIST
	elif dataset_name == "fashion-mnist":
		mean, std, channels, classes = (0.2860,), (0.3530,), 1, 10
		ds_class = datasets.FashionMNIST
	elif dataset_name == "cifar10":
		mean, std, channels, classes = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 3, 10
		ds_class = datasets.CIFAR10
	else:
		raise ValueError(f"Unsupported dataset: {dataset_name}")

	train_transform = transforms.Compose(
		[
			#transforms.RandomCrop(32, padding=4) if channels == 3 else transforms.Resize((28, 28)),
			#transforms.RandomHorizontalFlip() if channels == 3 else transforms.Lambda(lambda x: x),
			transforms.ToTensor(),
			transforms.Lambda(lambda x: torch.flatten(x))
			#transforms.Normalize(mean, std),
		]
	)
	test_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

	train_dataset = ds_class(root=data_dir, train=True, download=True, transform=train_transform)
	test_dataset = ds_class(root=data_dir, train=False, download=True, transform=test_transform)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

	input_dim = train_dataset[0][0].numel()
	return train_loader, test_loader, input_dim, classes


def train_one_epoch(model, loader, optimizer, criterion, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()
		clip_weights(model)

		running_loss += loss.item() * targets.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == targets).sum().item()
		total += targets.size(0)

	return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, targets)

		running_loss += loss.item() * targets.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == targets).sum().item()
		total += targets.size(0)

	return running_loss / total, 100.0 * correct / total


class BinaryVGG(nn.Module):
	def __init__(self, in_channels: int, num_classes: int, base_channels: int = 128):
		super().__init__()
		c1 = base_channels
		c2 = base_channels * 2
		c3 = base_channels * 4
		self.features = nn.Sequential(
			# Block 1
			BinaryConv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c1),
			BinaryActivation(),

			BinaryConv2d(c1, c1, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c1),
			BinaryActivation(),
			nn.MaxPool2d(2),                        # → 128 × 16 × 16
			# Block 2
			BinaryConv2d(c1, c2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c2),
			BinaryActivation(),

			BinaryConv2d(c2, c2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c2),
			BinaryActivation(),
			nn.MaxPool2d(2),                        # → 256 × 8 × 8
			# Block 3
			BinaryConv2d(c2, c3, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c3),
			BinaryActivation(),
			nn.MaxPool2d(2),                        # → 512 × 4 × 4
			nn.AdaptiveAvgPool2d((4, 4)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			BinaryLinear(c3 * 4 * 4, 1024, bias=False),
			nn.BatchNorm1d(1024),
			BinaryActivation(),
			BinaryLinear(1024, num_classes),
			nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(dim=1)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.classifier(self.features(x))


def main():
	parser = argparse.ArgumentParser(description="PyTorch Binary Neural Network")
	parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist", "cifar10"])
	parser.add_argument("--data-dir", type=str, default="./data")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	set_seed(args.seed)
	device = torch.device(args.device)

	train_loader, test_loader, input_dim, classes = get_dataloaders(
		args.dataset,
		args.data_dir,
		args.batch_size,
	)
	model = BinaryVGG(in_channels=1 if args.dataset in ["mnist", "fashion-mnist"] else 3, num_classes=classes).to(device)
	model = MLP(input_dim=input_dim, num_classes=classes).to(device)
	print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")
	for module in model.modules():
		if isinstance(module, (BinaryLinear, BinaryConv2d)):
			print(module.weight.shape)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

	print(model)
	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)
		scheduler.step()

		print(
			f"Epoch {epoch:02d}/{args.epochs} | "
			f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
			f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
		)


if __name__ == "__main__":
	main()
