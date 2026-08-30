from cifar_resnet_dwn import DWNResNetCIFAR, DWNResNetCIFAR2, DWNResNetCIFAREnsemble
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import torch_dwn as dwn
from tqdm import tqdm
import os

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if os.name == "nt":
            raise RuntimeError(
                "Distributed FSDP with CUDA requires NCCL, which is not available on native Windows. "
                "Use Linux/WSL2 for multi-GPU or multi-node runs."
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def prepare_model_for_fsdp(module):
    converted_names = []
    for module_name, submodule in module.named_modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            is_float_or_complex = param.dtype.is_floating_point or param.dtype.is_complex
            if is_float_or_complex:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param.requires_grad:
                raise RuntimeError(
                    f"FSDP cannot flatten trainable non-floating parameter: {full_name} ({param.dtype})."
                )

            tensor_value = param.detach()

            # Remove the existing parameter attribute so the same name can be reused as a buffer.
            if param_name in submodule._parameters:
                submodule._parameters.pop(param_name)
            if hasattr(submodule, param_name):
                delattr(submodule, param_name)

            # persistent=False: these are deterministic per-rank (computed from constructor
            # args), so they don't need to be synced across ranks or saved in checkpoints.
            submodule.register_buffer(param_name, tensor_value, persistent=False)
            converted_names.append(full_name)

    return converted_names


is_distributed, rank, world_size, local_rank = setup_distributed()
print(f"Distributed setup: is_distributed={is_distributed}, rank={rank}, world_size={world_size}, local_rank={local_rank}")
is_main_process = rank == 0

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script.")

device = torch.device(f"cuda:{local_rank}")
# Load Data
train_transform = transforms.Compose([
    #transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: torch.flatten(x))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: torch.flatten(x))
])

#epoch 30
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

#Epoch 50
# train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
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

binarized_train_dataset = TensorDataset(x_train, y_train)
binarized_test_dataset = TensorDataset(x_test, y_test)

if is_distributed:
    train_sampler = DistributedSampler(
        binarized_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    test_sampler = DistributedSampler(
        binarized_test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
else:
    train_sampler = None
    test_sampler = None

train_loader = DataLoader(
    dataset=binarized_train_dataset,
    batch_size=512,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=binarized_test_dataset,
    batch_size=512,
    shuffle=False,
    sampler=test_sampler,
    pin_memory=True,
)


# out_dim1 = 1 + (32 - 6)//2 
# out_dim2 = 1 + (out_dim1 -6)//2
# out_dim3 = 1 + (out_dim2 -3)//2
# kernel1 = 12
# mlp_layer = out_dim3 * out_dim3 * kernel1 * 4  #out_dim * out_dim * kernel1
# model = nn.Sequential(
#     dwn.DWNConvLayer(in_channels=3*therm_bits, groups=3, kernels=kernel1, depth=2, stride=2, receptive_field=6, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1, groups=4, kernels=kernel1*2, depth=2, stride=2, receptive_field=6, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1 * 2, groups=4 * 2, kernels=kernel1*4, depth=1, stride=2, receptive_field=3, flatten_output=True),
#     dwn.LUTLayer(mlp_layer, 1000, n=4),
#     #dwn.LUTLayer(int(mlp_layer / 4), 1000, n=4),
#     dwn.GroupSum(k=10, tau=1/0.1)
# )


# out_dim1 = 1 + (32 - 3)
# out_dim2 = 1 + (out_dim1 -3)
# out_dim3 = 1 + (out_dim2 -3)//2
# out_dim4 = 1 + (out_dim3 -3)//2
# out_dim5 = 1 + (out_dim4 -3)//2
# kernel1 = 64
# groupsNb = kernel1 // 4
# increase_factor = 4
# mlp_layer = out_dim5 * out_dim5 * kernel1 * (increase_factor ** 4)  #out_dim * out_dim * kernel1
# tau = (mlp_layer  / 10) / 100
# model = nn.Sequential(
#     dwn.DWNConvLayer(in_channels=3*therm_bits, groups=3, kernels=kernel1, depth=1, stride=1, receptive_field=3, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1, groups=groupsNb, kernels=kernel1 * (increase_factor ** 1), depth=1, stride=1, receptive_field=3, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1 * (increase_factor ** 1), groups=groupsNb * (increase_factor ** 1), kernels=kernel1 * (increase_factor ** 2), depth=1, stride=2, receptive_field=3, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1 * (increase_factor ** 2), groups=groupsNb * (increase_factor ** 2), kernels=kernel1 * (increase_factor ** 3), depth=1, stride=2, receptive_field=3, flatten_output=False),
#     dwn.DWNConvLayer(in_channels=kernel1 * (increase_factor ** 3), groups=groupsNb * (increase_factor ** 3), kernels=kernel1 * (increase_factor ** 4), depth=1, stride=2, receptive_field=3, flatten_output=True),
#     dwn.LUTLayer(mlp_layer, mlp_layer * 2, n=4),
#     dwn.LUTLayer(mlp_layer * 2, mlp_layer, n=4),
#     dwn.GroupSum(k=10, tau=tau)
# )

model = DWNResNetCIFAREnsemble()


# mlp_layer = out_dim1 * out_dim1 * kernel1
# model = nn.Sequential(
#     dwn.DWNConvLayer(in_channels=3*therm_bits, groups=3, kernels=kernel1, flatten_output=True, learnable_connections=True),
#     dwn.LUTLayer(mlp_layer, int(mlp_layer / 4), n=4),
#     dwn.LUTLayer(int(mlp_layer / 4), 1000, n=4),
#     dwn.GroupSum(k=10, tau=1/0.1)
# )

model = model.to(device)

use_fsdp = is_distributed and world_size > 1
if use_fsdp:
    if is_main_process:
        print(f"Using FSDP across {world_size} processes")

    converted_param_names = prepare_model_for_fsdp(model)
    if converted_param_names and is_main_process:
        converted_module_names = sorted(
            {
                name.rsplit(".", 1)[0] if "." in name else "<root>"
                for name in converted_param_names
            }
        )
        print(
            f"Converted {len(converted_param_names)} non-floating Parameters to buffers for FSDP compatibility."
        )
        print("Modules with converted non-floating Parameters:")
        for module_name in converted_module_names:
            print(f"  - {module_name}")

    # Wrap at coarse granularity: only submodules with >= 50M params get their own
    # FSDP unit. Everything else stays in the root unit. Fewer FSDP units means
    # fewer all-gather/reduce-scatter calls per forward/backward pass.
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=50_000_000,
    )
    # Reduce gradients in float16 to halve inter-GPU communication bandwidth.
    # Parameters and buffers stay in float32 (required by DWN CUDA kernels).
    mixed_precision = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float32,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
        sync_module_states=True,
        use_orig_params=True,
    )
elif is_main_process:
    print("Running without FSDP (single process).")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            pred = model(batch_x).argmax(dim=1)
            correct += (pred == batch_y).sum()
            total += batch_y.size(0)

        if is_distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        acc = (correct.float() / total.float()).item()
    return acc

def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs):
    progress_bar = tqdm(range(epochs), desc="Training Progress", disable=not is_main_process)

    for epoch in progress_bar:
        model.train()
        if is_distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        correct_train = torch.tensor(0, device=device)
        total_train = torch.tensor(0, device=device)

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()

            pred_train = outputs.argmax(dim=1)

            correct_train += (pred_train == batch_y).sum()
            total_train += batch_y.size(0)

        if is_distributed:
            dist.all_reduce(correct_train, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_train, op=dist.ReduceOp.SUM)

        train_acc = (correct_train.float() / total_train.float()).item()

        #scheduler.step()

        if epoch % 10 == 0:
            test_acc = evaluate(model, test_loader)
            if is_main_process:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=150)

if use_fsdp:
    if is_main_process:
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(".", "cifar_model.pt"))
else:
    if is_main_process:
        torch.save(model.state_dict(), os.path.join(".", "cifar_model.pt"))

if is_distributed:
    dist.barrier()
    dist.destroy_process_group()

"""
First one using the 5 layer of CNN with kernel = 32
Training Progress:   0%|                                                                                        | 0/150 [00:00<?, ?it/s]Epoch 1/150, Train Loss: 1.7863, Train Accuracy: 0.3085, Test Accuracy: 0.3722
Training Progress:   7%|█████▏                                                                       | 10/150 [08:30<1:58:33, 50.81s/it]Epoch 11/150, Train Loss: 0.4358, Train Accuracy: 0.9608, Test Accuracy: 0.5021
Training Progress:  13%|██████████▎                                                                  | 20/150 [17:04<1:51:27, 51.45s/it]Epoch 21/150, Train Loss: 0.3007, Train Accuracy: 0.9928, Test Accuracy: 0.5066
Training Progress:  20%|███████████████▍                                                             | 30/150 [25:37<1:41:50, 50.92s/it]Epoch 31/150, Train Loss: 0.2763, Train Accuracy: 0.9931, Test Accuracy: 0.5127
Training Progress:  27%|████████████████████▌                                                        | 40/150 [34:06<1:33:11, 50.83s/it]Epoch 41/150, Train Loss: 0.3688, Train Accuracy: 0.9733, Test Accuracy: 0.5013
Training Progress:  29%|██████████████████████                                                       | 43/150 [36:39<1:30:54, 50.98s/it]| 43/150 [37:24<1:33:05, 52.20s/it]

This one with the one with 3 layer of CNN with kernel = 12
Training Progress:   0%|                                                                                        | 0/150 [00:00<?, ?it/s]Epoch 1/150, Train Loss: 2.0185, Train Accuracy: 0.2191, Test Accuracy: 0.2889
Training Progress:   7%|█████▎                                                                         | 10/150 [00:29<06:39,  2.85s/it]Epoch 11/150, Train Loss: 1.5569, Train Accuracy: 0.4237, Test Accuracy: 0.4158
Training Progress:  13%|██████████▌                                                                    | 20/150 [00:56<05:56,  2.74s/it]Epoch 21/150, Train Loss: 1.5928, Train Accuracy: 0.4360, Test Accuracy: 0.4247
Training Progress:  20%|███████████████▊                                                               | 30/150 [01:24<05:27,  2.73s/it]Epoch 31/150, Train Loss: 1.5583, Train Accuracy: 0.4408, Test Accuracy: 0.4248
Training Progress:  27%|█████████████████████                                                          | 40/150 [01:51<04:57,  2.70s/it]Epoch 41/150, Train Loss: 1.5086, Train Accuracy: 0.4393, Test Accuracy: 0.4189
Training Progress:  33%|██████████████████████████▎                                                    | 50/150 [02:18<04:29,  2.70s/it]Epoch 51/150, Train Loss: 1.5279, Train Accuracy: 0.4421, Test Accuracy: 0.4275
Training Progress:  40%|███████████████████████████████▌                                               | 60/150 [02:47<04:05,  2.73s/it]Epoch 61/150, Train Loss: 1.6308, Train Accuracy: 0.4434, Test Accuracy: 0.4263
Training Progress:  47%|████████████████████████████████████▊                                          | 70/150 [03:14<03:36,  2.71s/it]Epoch 71/150, Train Loss: 1.5037, Train Accuracy: 0.4448, Test Accuracy: 0.4271
Training Progress:  53%|██████████████████████████████████████████▏                                    | 80/150 [03:41<03:07,  2.68s/it]Epoch 81/150, Train Loss: 1.5314, Train Accuracy: 0.4441, Test Accuracy: 0.4327
Training Progress:  60%|███████████████████████████████████████████████▍                               | 90/150 [04:08<02:40,  2.67s/it]Epoch 91/150, Train Loss: 1.5057, Train Accuracy: 0.4443, Test Accuracy: 0.4245
Training Progress:  67%|████████████████████████████████████████████████████                          | 100/150 [04:36<02:15,  2.72s/it]Epoch 101/150, Train Loss: 1.5575, Train Accuracy: 0.4468, Test Accuracy: 0.4282
Training Progress:  73%|█████████████████████████████████████████████████████████▏                    | 110/150 [05:03<01:48,  2.70s/it]Epoch 111/150, Train Loss: 1.4928, Train Accuracy: 0.4450, Test Accuracy: 0.4271
Training Progress:  80%|██████████████████████████████████████████████████████████████▍               | 120/150 [05:31<01:21,  2.73s/it]Epoch 121/150, Train Loss: 1.5026, Train Accuracy: 0.4442, Test Accuracy: 0.4286
Training Progress:  86%|███████████████████████████████████████████████████████████████████           | 129/150 [05:56<00:57,  2.72s/it]^Training Progress:  86%|███████████████████████████████████████████████████████████████████           | 129/150 [05:57<00:58,  2.77s/it]

5 layers with k=64
Epoch 1/150, Train Loss: 1.7529, Train Accuracy: 0.3365, Test Accuracy: 0.4074
Training Progress:   7%|███▉                                                       | 10/150 [18:02<4:11:57, 107.98s/it]Epoch 11/150, Train Loss: 0.4552, Train Accuracy: 0.9761, Test Accuracy: 0.5200
Training Progress:  13%|███████▊                                                   | 20/150 [36:01<3:53:30, 107.77s/it]Epoch 21/150, Train Loss: 0.3103, Train Accuracy: 0.9934, Test Accuracy: 0.5315
Training Progress:  18%|██████████▌                                                | 27/150 [48:38<3:41:03, 107.83s/it]

Notice the train accuracy in the first one is almost 100 while in second one it is around 44
There seems to be not much improvement with increasing Kernel number

DWNResnetCIFAR2 with 5 layers and kernel=32 with only 1 residual connection
Training Progress:   0% 0/150 [00:00<?, ?it/s]Epoch 1/150, Train Loss: 1.7948, Train Accuracy: 0.2803, Test Accuracy: 0.3622
Training Progress:   7% 10/150 [09:45<2:16:25, 58.47s/it]Epoch 11/150, Train Loss: 0.9427, Train Accuracy: 0.7504, Test Accuracy: 0.5369
Training Progress:  13% 20/150 [19:31<2:06:47, 58.52s/it]Epoch 21/150, Train Loss: 0.8384, Train Accuracy: 0.7974, Test Accuracy: 0.5419
Training Progress:  20% 30/150 [29:17<1:56:59, 58.50s/it]Epoch 31/150, Train Loss: 0.8758, Train Accuracy: 0.7920, Test Accuracy: 0.5400
Training Progress:  27% 40/150 [39:03<1:47:19, 58.54s/it]Epoch 41/150, Train Loss: 0.8741, Train Accuracy: 0.7856, Test Accuracy: 0.5333
Training Progress:  33% 50/150 [48:50<1:37:33, 58.54s/it]Epoch 51/150, Train Loss: 0.9635, Train Accuracy: 0.7698, Test Accuracy: 0.5269
Training Progress:  34% 51/150 [50:27<1:37:57, 59.37s/it]

The performance does improve by like 5% compared to without residual connection but not enough


"""