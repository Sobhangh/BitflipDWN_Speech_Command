import gc
import sys
sys.path.insert(0, './BitflipDWN_Speech_Command')
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch_dwn as dwn
from tqdm import tqdm
import random
import os
from types import SimpleNamespace
from dsprites import CountingShapes, OrientationShapes
import copy
import pandas as pd


device = "cuda"
BATCH_SIZE = 128
scheduler = None


def hardend_model(model):
  lf_model = copy.deepcopy(model)
  for layer_idx, layer in enumerate(lf_model.net):
        if isinstance(layer, (BinaryLinear, BinaryConv2d)):
          layer.weight.data = ((layer.weight.data > 0).float() * 2) - 1
  return lf_model

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
	def __init__(self, input_dim: int, num_classes: int, width = 100):
		super().__init__()
		self.net = nn.Sequential(
			nn.Flatten(),
			BinaryLinear(input_dim, width, bias=False),
			nn.BatchNorm1d(width),
			BinaryActivation(),
			#nn.Dropout(0.2),

			BinaryLinear(width, width, bias=False),
			nn.BatchNorm1d(width),
			BinaryActivation(),


			# nn.BatchNorm1d(width),
			# BinaryActivation(),
			# BinaryLinear(width, width, bias=False),

			# BinaryLinear(width, width, bias=False),
			# nn.BatchNorm1d(width),
			# BinaryActivation(),
			# #nn.Dropout(0.2),

			BinaryLinear(width, num_classes),
			#This batchnorm effect the bit flip very negatively by making the gradient very smal
			#nn.BatchNorm1d(num_classes, affine=False),
      #nn.LogSoftmax(dim=1)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


@torch.no_grad()
def clip_weights(model: nn.Module, min_val: float = -1.0, max_val: float = 1.0) -> None:
	for module in model.modules():
		if isinstance(module, (BinaryLinear, BinaryConv2d)):
			module.weight.clamp_(min_val, max_val)

class BinaryVGG(nn.Module):
	def __init__(self, in_channels: int, num_classes: int, base_channels: int = 128):
		super().__init__()
		c1 = base_channels
		c2 = base_channels * 2
		c3 = base_channels * 4
		self.net = nn.Sequential(
			#features part
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

			#classifier part
			nn.Flatten(),
			BinaryLinear(c3 * 4 * 4, 1024, bias=False),
			nn.BatchNorm1d(1024),
			BinaryActivation(),
			BinaryLinear(1024, num_classes),
			#nn.BatchNorm1d(num_classes, affine=False),
      #nn.LogSoftmax(dim=1)
		)
		# self.classifier = nn.Sequential(
		# 	nn.Flatten(),
		# 	BinaryLinear(c3 * 4 * 4, 1024, bias=False),
		# 	nn.BatchNorm1d(1024),
		# 	BinaryActivation(),
		# 	BinaryLinear(1024, num_classes),
		# 	nn.BatchNorm1d(num_classes, affine=False),
    #   nn.LogSoftmax(dim=1)
		# )


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		#return self.classifier(self.features(x))
		return self.net(x)

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
            clip_weights(model)

            pred_train = outputs.argmax(dim=1)

            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_acc = correct_train / total_train

        scheduler.step()

        if epoch % 10 == 0:
          test_acc = evaluate(model, x_test, y_test)
          print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')


experiment_comb = {
    'mnist': 100,
    'fmnist': 100,
    'counting': 32,
    'orientation': 32,
    'orientation-2obj': 32,
}

for k,v in experiment_comb.items():
    layer_size = v
    args = {
        'task': k,
        'use_wandb': False, # Default for action='store_true' is False if not present
        'seed': 453,
        'batch_size': BATCH_SIZE,
        'layer_size': layer_size,
        'lr': 0.001,
        'step_size': 24,
        'epochs': 50
    }
    args = SimpleNamespace(**args)

    print("\n**********************************************************************")
    print(f"Training started for {args.task} model with layer size {args.layer_size} \n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    all_data = None
    classes = 10
    if args.task == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = train_dataset[0][0].numel()
    elif args.task == 'fmnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = train_dataset[0][0].numel()
    else:
        classes = 3
        resolution=(64, 64)
        out_size = args.layer_size
        transformation = transforms.Compose([
            transforms.Resize(resolution),
            #transforms.Lambda(lambda x: x.flatten())
        ])

        match args.task:
            case 'counting':
                cls = CountingShapes
            case 'orientation' | 'orientation-2obj':
                cls = OrientationShapes
            case _:
                raise ValueError('Task is not implemented!')

        all_data = cls.from_path(f'{args.task}-o3-128.npz', transforms=transformation)

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [0.9, 0.1], generator=generator)

        input_dim = 3 * resolution[0] * resolution[1]
        print(f"Size of the test dataset {len(train_dataset)}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    if k in ['counting', 'orientation', 'orientation-2obj']:
        model = BinaryVGG(in_channels=3, num_classes=classes, base_channels=args.layer_size).to(device)
    else:
        model = MLP(input_dim=input_dim, num_classes=classes, width=args.layer_size).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=args.step_size)

    save_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(save_dir, exist_ok=True)
    if args.task in ['counting', 'orientation', 'orientation-2obj']:
        model_path = os.path.join(save_dir, f"{args.task}_ConvBNN_model_{args.layer_size}.pt")
    else:
        model_path = os.path.join(save_dir, f"{args.task}_BNN_model_{args.layer_size}.pt")
        
    train_and_evaluate(model, optimizer, x_train, y_train, x_test, y_test, epochs=args.epochs, batch_size=BATCH_SIZE)
    
    torch.save(model.state_dict(), model_path) 

    print("++++++++++++++++++ STARTING UNTARGETED BIT FLIP ATTACKS ++++++++++++++++++++++")

    data_name = args.task
    results_df = pd.DataFrame(columns=['batch_size', 'trial_number', 'base_acc', 'nb_flips'])
    base_acc = evaluate(model, x_test, y_test)
    print(f"base acc: {base_acc}")
    n_samples = x_test.shape[0]
    print(f"number of test data: {n_samples}")
    for batch_size in [256]: # 32,64,128,256,
        #print(f"****************** New Batch size {batch_size} **********************")
        for trial in range(5):
            print("****** new trial ********")
            permutation = torch.randperm(n_samples)
            indices = permutation[:batch_size]
            batch_x = x_test[indices].cuda(device)
            batch_y = y_test[indices].cuda(device)
            lf_model = hardend_model(model) #copy.deepcopy(model)
            acc = evaluate(lf_model, x_test, y_test)
            base_acc = acc
            print(f"base accuracy: {acc}")
            nb_flips = 0
            acc_hist = [acc]
            layer_hist = []
            loss_hist = []
            while acc > 0.01 + 1/classes :
                lf_model.train()
                lf_model.zero_grad()
                outputs = lf_model(batch_x)
                loss = cross_entropy(outputs, batch_y)
                loss.backward()
                loss_hist.append(loss.item())
                lf_model.eval()
                best_gradients = []
                for layer_idx, layer in enumerate(lf_model.net):
                    if isinstance(layer, (BinaryLinear, BinaryConv2d)):
                        grad_copy = torch.abs(layer.weight.grad.clone())
                        while True:
                            max_flat_index = torch.argmax(grad_copy)
                            #max_flat_index = torch.randint(0, grad_copy.shape.prod(), (1,)).item()
                            max_idx = torch.unravel_index(max_flat_index, layer.weight.grad.shape)
                            previous_weight = layer.weight.data[max_idx]
                            b = (previous_weight > 0)
                            #Gradient ascent
                            m = b ^ (layer.weight.grad[max_idx] > 0)
                            if not m:
                                grad_copy[max_idx] = -torch.inf
                            else:
                                b_hat = (float(b ^ m) * 2) - 1
                                lf_model.net[layer_idx].weight.data[max_idx] = b_hat
                                outputs = lf_model(batch_x)
                                flipped_loss = cross_entropy(outputs, batch_y).item()
                                best_gradients.append((layer_idx, max_idx, flipped_loss, b_hat))
                                lf_model.net[layer_idx].weight.data[max_idx] = previous_weight
                                break
                        

                crossl = max(best_gradients, key=lambda x: x[2])
                lf_model.net[crossl[0]].weight.data[crossl[1]] = crossl[3]
                nb_flips += 1
                curr_acc = evaluate(lf_model, x_test, y_test)
                acc_hist.append(curr_acc)
                layer_hist.append(crossl[0])
                drop_acc = acc - curr_acc
                acc = curr_acc
                print(f"Batch size {batch_size}, trial {trial}; Bit flipped total {nb_flips}, loss {crossl[3]},{drop_acc}; accuracy {acc}, layer {crossl[0]}")
            outputs = lf_model(batch_x)
            loss_hist.append(cross_entropy(outputs, batch_y).item())
            new_row_data = {
                'batch_size': batch_size,
                'trial_number': trial,
                'base_acc': base_acc,
                'nb_flips': nb_flips
            }
            results_df.loc[len(results_df)] = new_row_data
            del lf_model
            del batch_x, batch_y
            del acc_hist, layer_hist, loss_hist
            gc.collect()
            torch.cuda.empty_cache()
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)
    if args.task in ['counting', 'orientation', 'orientation-2obj']:
        res_path = os.path.join(save_dir,f"bit_flip_untargeted_ConvBNN_{data_name}_{args.layer_size}.csv")
    else:
        res_path = os.path.join(save_dir,f"bit_flip_untargeted_BNN_{data_name}_{args.layer_size}.csv")
    results_df.to_csv(res_path, index=False)
    print("Untargeted bit flip statistics saved!!")

    if args.task in ['counting', 'orientation', 'orientation-2obj']:
        print("SKIPPING TARGETED BIT FLIP ATTACKS FOR OOD DATASETS")
        continue
    print("++++++++++++++++++ STARTING TARGETED BIT FLIP ATTACKS ++++++++++++++++++++++")

    del results_df
    batch_size = 256
    data_name = args.task
    results_df = pd.DataFrame(columns=['source', 'target_class', 'level', 'trial_number', 'base_acc', 'nb_flips', 'last_asr', 'last_ta'])
    for level in range(1,4):
        for target_class in range(0, classes):
            for source in (range(0, classes) if level != 1 else [-1]):
                if source != target_class:
                    print("**************************************************")
                    print(f"source {source}, target {target_class}, level {level}")
                    for trial in range(5):
                        print("****** new trial ********")
                        asr_hist = []
                        layer_hist = []
                        loss_hist = []
                        ta_hist = []
                        if level != 1:
                            #source = 3
                            x_test_source = x_test[y_test == source]
                            perm = torch.randperm(x_test_source.shape[0])
                            x_test_source_search = x_test_source[perm[:x_test_source.shape[0]//2]]
                            x_test_source_eval = x_test_source[perm[x_test_source.shape[0]//2:]]
                            print(f"size of source class {x_test_source.shape}")
                            x_test_rest = x_test[y_test != source]
                            y_test_rest = y_test[y_test != source]
                            if level == 3:
                                rest_data_size = x_test_rest.shape[0]
                                perm = torch.randperm(rest_data_size)
                                rest_search_idx = perm[:x_test_source.shape[0]//2]
                                x_test_rest_search = x_test_rest[rest_search_idx]
                                y_test_rest_search = y_test_rest[rest_search_idx]
                                rest_eval_idx = perm[x_test_source.shape[0]//2:]
                                x_test_rest_eval = x_test_rest[rest_eval_idx]
                                y_test_rest_eval = y_test_rest[rest_eval_idx]
                                correct_idx = ~torch.isin(torch.arange(rest_data_size), rest_search_idx)
                                print(f"size of rest  {x_test_rest.shape}")
                                print(f"correct TA rest elems {correct_idx.sum().item()}")

                        lf_model = hardend_model(model) #copy.deepcopy(model)
                        base_acc = evaluate(lf_model, x_test, y_test)
                        print(f"base acc: {base_acc}")
                        acc = base_acc
                        #THIS IS USED FOR THE BATCH SIZE IN THE EXPERIMENTS FOR LEVEL 1
                        batch_size = 32
                        nb_flips = 0
                        n_samples = x_test.shape[0]
                        print(f"number of test data: {n_samples}")


                        asr_history = [-1] * 20
                        #Stop should be used only once in a loop as it is stateful; TO DO: Make it stateless
                        def stop_condition():
                            asr_threshold = base_acc #0.99
                            if level == 1:
                                acc_val = evaluate(lf_model, x_test, torch.full((n_samples,),target_class))
                                test_acc = None
                            elif level == 2:
                                acc_val = evaluate(lf_model, x_test_source_eval, torch.full((x_test_source_eval.shape[0],),target_class))
                                test_acc = evaluate(lf_model, x_test_rest, y_test_rest)
                            else:
                                acc_val = evaluate(lf_model, x_test_source_eval, torch.full((x_test_source_eval.shape[0],),target_class))
                                correct_idx = ~torch.isin(torch.arange(rest_data_size), rest_search_idx)
                                test_acc = evaluate(lf_model, x_test_rest[correct_idx], y_test_rest[correct_idx])
                            asr_history.append(acc_val)
                            asr_history.pop(0)
                            if acc_val == 0:
                                stable = False
                            else:
                                stable = sum([asr_history[i] == acc_val for i in range(len(asr_history))]) == len(asr_history)
                            return acc_val < asr_threshold and not stable, acc_val, test_acc

                        def get_loss():
                            if level == 1:
                                perm1 = torch.randperm(x_test.shape[0])
                                batch_x  = x_test[perm1[:batch_size]].cuda(device)
                                batch_y = torch.full((batch_x.shape[0],),target_class).cuda(device)
                                outputs = lf_model(batch_x)
                                return cross_entropy(outputs, batch_y)
                            elif level == 2:
                                batch_x, batch_y = x_test_source_search.cuda(device), torch.full((x_test_source_search.shape[0],),target_class).cuda(device)
                                outputs = lf_model(batch_x)
                                return cross_entropy(outputs, batch_y)
                            else:
                                batch_x1, batch_y1 = x_test_source_search.cuda(device), torch.full((x_test_source_search.shape[0],),target_class).cuda(device)
                                batch_x2, batch_y2 = x_test_rest_search.cuda(device), y_test_rest_search.cuda(device)
                                outputs = lf_model(torch.cat((batch_x1,batch_x2), dim=0))
                                return cross_entropy(outputs, torch.cat((batch_y1,batch_y2), dim=0))
                            


                        continue_tbf = True
                        while continue_tbf:
                            lf_model.train()
                            correct_train = 0
                            total_train = 0
                            lf_model.zero_grad()
                            loss = get_loss()
                            loss.backward()
                            loss_hist.append(loss.item())

                            lf_model.eval()
                            best_gradients = []
                            for layer_idx, layer in enumerate(lf_model.net):
                                if isinstance(layer, (BinaryLinear, BinaryConv2d)):
                                    if layer.weight.grad is None:
                                        continue # Skip if no gradients for this layer

                                    grad_copy = torch.abs(layer.weight.grad.clone())
                                    while True:
                                        if grad_copy.max() == -torch.inf:
                                            break # All values have been exhausted/marked as inf

                                        max_flat_index = torch.argmax(grad_copy)
                                        max_idx = torch.unravel_index(max_flat_index, layer.weight.grad.shape)
                                        previous_weight = layer.weight.data[max_idx]
                                        b = (previous_weight > 0)
                                        
                                        #Gradient descent
                                        m = b ^ (layer.weight.grad[max_idx] < 0)
                                        
                                        if not m:
                                            grad_copy[max_idx] = -torch.inf
                                        else:
                                            b_hat = (float(b ^ m) * 2) - 1

                                            with torch.no_grad(): # Added no_grad context
                                                lf_model.net[layer_idx].weight.data[max_idx] = b_hat

                                            best_gradients.append((layer_idx, max_idx, get_loss().item(), b_hat))

                                            with torch.no_grad(): # Added no_grad context
                                                lf_model.net[layer_idx].weight.data[max_idx] = previous_weight
                                            break

                            if not best_gradients: # Added check for empty best_gradients
                                print("No suitable bit flips found to decrease loss. Breaking loop.")
                                break

                            crossl = min(best_gradients, key=lambda x: x[2])
                            with torch.no_grad(): # Added no_grad context
                                lf_model.net[crossl[0]].weight.data[crossl[1]] = crossl[3]
                            nb_flips += 1
                            stop = stop_condition()
                            acc_values = stop[1]
                            continue_tbf = stop[0]
                            asr_hist.append(acc_values)
                            layer_hist.append(crossl[0])
                            ta_hist.append(stop[2])
                            print(f"Bit flipped total {nb_flips}, loss {crossl[2]}; asr {acc_values}, ta {stop[2]}, layer {crossl[0]}")

                        loss_hist.append(get_loss())
                        new_row_data = {
                            'source': source,
                            'target_class': target_class,
                            'level': level,
                            'trial_number': trial,
                            'base_acc': base_acc,
                            'nb_flips': nb_flips,
                            'last_asr': asr_hist[-1],
                            'last_ta' : ta_hist[-1],
                        }
                        results_df.loc[len(results_df)] = new_row_data
                        del lf_model
                        del asr_hist, layer_hist, loss_hist, ta_hist
                        if level != 1:
                            del x_test_source, x_test_source_search, x_test_source_eval
                            del x_test_rest, y_test_rest
                            if level == 3:
                                del x_test_rest_search, y_test_rest_search
                                del x_test_rest_eval, y_test_rest_eval
                        gc.collect()
                        torch.cuda.empty_cache()

    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)
    if args.task in ['counting', 'orientation', 'orientation-2obj']:
        res_path = os.path.join(save_dir, f"bit_flip_targeted_ConvBNN_{data_name}_{args.layer_size}.csv")
    else:
        res_path = os.path.join(save_dir, f"bit_flip_targeted_BNN_{data_name}_{args.layer_size}.csv")
    results_df.to_csv(res_path, index=False)
    print("Untargeted stats saved!!")

    del model, optimizer, scheduler
    del train_loader, test_loader, test_dataset, train_dataset
    del all_data
    del x_train, y_train, x_test, y_test
    del results_df
    gc.collect()
    torch.cuda.empty_cache()
    print("Model and data deleted, cache cleared, moving to next experiment")







                