import argparse
import os

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch_dwn as dwn

from dsprites import CountingShapes, OrientationShapes, PairedShapes
from tqdm import tqdm
import wandb

def train_and_evaluate(model: nn.Module, optimizer, scheduler, 
                       tr_loader: DataLoader, val_loader: DataLoader, device: torch.device, 
                       epochs:int, run:wandb.Run | None = None):
    
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(tr_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for (panels, target) in train_bar:
            optimizer.zero_grad()
            
            x_tr = panels.to(device)
            y_tr = target.to(device)
            
            pred = model(x_tr)
            loss = cross_entropy(pred, y_tr)
            loss.backward()
            optimizer.step()
            
            pred_train = pred.argmax(dim=1)
            correct_train += (pred_train == y_tr).sum().item()
            total_train += y_tr.size(0)
        
        train_acc = (correct_train / total_train) * 100
        
        scheduler.step()
        
        # eval mode
        model.eval()
        with torch.no_grad():
            correct_test = 0
            total_test = 0
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ', leave=False)
            for (panels, target) in val_bar:
                x_val = panels.to(device)
                pred = (model(x_val).cpu()).argmax(dim=1).numpy()
                correct_test += (pred == target.numpy()).sum()
                total_test += target.size(0)
            test_acc = (correct_test / total_test) * 100
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        if run is not None:
            run.log({
                    'epoch': epoch + 1,
                    'avg_train_acc': train_acc,
                    'avg_val_acc': test_acc,
                })
    print('Training finished!')
    if run is not None:
        run.finish()

if __name__ == '__main__':
    # Load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='counting', choices=['counting', 'orientation', 'paired', 'orientation-2obj'], help='Task to run')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for reproductability (default 123)')
    parser.add_argument('--batch-size', '-bs', type=int, default=100, help='Batch size (default 100)')
    parser.add_argument('--layer-size', '-l', type=int, default=6000, help='First layer output size (default = 6000)')
    parser.add_argument('-n', type=int, default=6, help='LUT input size (default 6)')
    parser.add_argument('--tau-div', type=float, default=0.3, help='Tau dividend, adjust per class output size! (default 0.3)')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate (default 1e-2)')
    parser.add_argument('--step-size', type=int, default=10, help='Step scheduler step size (default 10)')
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs (default 50)')
    args = parser.parse_args()
    
    resolution=(64, 64)
    out_size = args.layer_size
    transformation = transforms.Compose([
        transforms.Resize(resolution),
        transforms.Lambda(lambda x: x.flatten())
    ])

    match args.task:
        case 'counting':
            cls = CountingShapes
        case 'orientation' | 'orientation-2obj':
            cls = OrientationShapes
        case 'paired':
            cls = PairedShapes
        case _:
            raise ValueError('Task is not implemented!')

    all_data = cls.from_path(f'{args.task}-o3-128.npz', transforms=transformation)

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(all_data, [0.9, 0.1], generator=generator)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    in_size = 3 * resolution[0] * resolution[1]
    tau = 1 / args.tau_div

    model = nn.Sequential(
        dwn.LUTLayer(in_size, out_size, n=args.n, mapping='learnable'),
        dwn.LUTLayer(out_size, out_size // 2, n=args.n),
        dwn.GroupSum(k=3, tau=tau)
    )

    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES')
    device = torch.device('cuda:{}'.format(visible_device))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=args.step_size)

    print('------Model------')
    print(model)
    
    if args.use_wandb:
        project = f'o3-{args.task}'
        wandb_conf = dict(
                        batch_size=args.batch_size,
                        n=args.n,
                        layer_size=out_size,
                        lr=args.lr,
                        step_size=args.step_size,
                        tau=tau,
                        epochs=args.epochs,
                        )
        run = wandb.init(project=project, config=wandb_conf)
        run.define_metric("epoch", hidden=True)
        run.define_metric("avg_train_acc", step_metric="epoch", summary="min,max")
        run.define_metric("avg_val_acc", step_metric="epoch", summary="min,max")
    else:
        run = None
    
    train_and_evaluate(model, optimizer, scheduler, train_loader, test_loader, device, epochs=args.epochs, run=run)