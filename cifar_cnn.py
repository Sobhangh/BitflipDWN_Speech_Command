import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_dwn as dwn
from tqdm import tqdm
import os

device = "cuda"
# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: torch.flatten(x))
])

#epoch 30
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

#Epoch 50
# train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

# Binarize with distributive thermometer
# thermometer = dwn.DistributiveThermometer(5).fit(x_train)
# x_train = thermometer.binarize(x_train).flatten(start_dim=1)
# x_test = thermometer.binarize(x_test).flatten(start_dim=1)

# thermometer = dwn.Thermometer(1)
# thermometer.thresholds = torch.tensor([0])
# #thermometer = dwn.thermometer.fit(x_train)
# x_train = thermometer.binarize(x_train).flatten(start_dim=1)
# x_test = thermometer.binarize(x_test).flatten(start_dim=1)

# model = nn.Sequential(
#     dwn.LUTLayer(x_train.size(1), 80000, n=2, mapping='learnable'),
#     dwn.LUTLayer(80000, 80000, n=2),
#     dwn.LUTLayer(80000, 80000, n=2),
#     dwn.LUTLayer(80000, 80000, n=2),
#     dwn.GroupSum(k=10, tau=1/0.1)
# )


# layer_size = 25000 #8000
# model = nn.Sequential(
#     dwn.LUTLayer(x_train.size(1), layer_size, n=4, mapping='learnable'),
#     dwn.LUTLayer(layer_size, layer_size, n=4),
#     dwn.LUTLayer(layer_size, layer_size, n=4),
#     dwn.LUTLayer(layer_size, layer_size, n=4),
#     dwn.LUTLayer(layer_size, 1000, n=4),
#     dwn.GroupSum(k=10, tau=1/0.1)
# )

# model = nn.Sequential(
#     dwn.LUTLayer(x_train.size(1), 2000, n=6, mapping='learnable'),
#     dwn.LUTLayer(2000, 1000, n=6),
#     dwn.GroupSum(k=10, tau=1/0.3)
# )

therm_bits=3
thermometer = dwn.DistributiveThermometer(num_bits=therm_bits, channel_wise=True).fit(x_train)
x_train = thermometer.binarize(x_train)
x_test = thermometer.binarize(x_test)
out_dim1 = 32 - 15 + 1
out_dim2 = 1 + (out_dim1 -6)//2
kernel1 = 12
mlp_layer = out_dim2 * out_dim2 * kernel1 * 2  #out_dim * out_dim * kernel1
# model = nn.Sequential(
#     dwn.DWNConvLayer(in_channels=3*therm_bits, groups=3, kernels=kernel1, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1, groups=4, kernels=kernel1*2, depth=2, stride=2, receptive_field=6),
#     dwn.LUTLayer(mlp_layer, 1000, n=4),
#     #dwn.LUTLayer(int(mlp_layer / 4), 1000, n=4),
#     dwn.GroupSum(k=10, tau=1/0.1)
# )
mlp_layer = out_dim1 * out_dim1 * kernel1
model = nn.Sequential(
    dwn.DWNConvLayer(in_channels=3*therm_bits, groups=3, kernels=kernel1, flatten_output=True, learnable_connections=True),
    dwn.LUTLayer(mlp_layer, int(mlp_layer / 4), n=4),
    dwn.LUTLayer(int(mlp_layer / 4), 1000, n=4),
    dwn.GroupSum(k=10, tau=1/0.1)
)

model = model.cuda()
if device.type == "cuda" and torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs for student actor modules")
		# Wrap actor branches only, so the top-level student API remains unchanged.
		model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, x_train, y_train, x_test, y_test, epochs, batch_size):
    n_samples = x_train.shape[0]

    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in progress_bar:
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0

        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)

            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()

            pred_train = outputs.argmax(dim=1)

            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_acc = correct_train / total_train

        #scheduler.step()

        if epoch % 10 == 0:
          test_acc = evaluate(model, x_test, y_test)
          print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

train_and_evaluate(model, optimizer, x_train, y_train, x_test, y_test, epochs=150, batch_size=128)
torch.save(model.state_dict(), os.path.join("/", "cifar_model.pt"))
