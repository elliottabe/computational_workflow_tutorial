import argparse
import torch
from pathlib import Path
from test_tube import Experiment
from omegaconf import OmegaConf

from utils import * 
from train import train, test
from models import NeuralNetwork

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Create paths if they don't exist and Path objects
    for k in cfg.paths.keys():
        if k != "user":
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    save_model = cfg.paths.ckpt_dir
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=cfg.dataset.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=cfg.dataset.batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    model = NeuralNetwork().to(device)
    if (save_model/'model.pth').exists():
        model.load_state_dict(torch.load(save_model/'model.pth'))
        print('loaded model from {}'.format(save_model/'model.pth'))
    print(model)

    # Define loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.dataset.lr)

    ##### Save config at start #####
    OmegaConf.save(config=cfg, f=cfg.paths.log_dir / 'run_config.yaml')

    # Train and test the model
    for t in range(cfg.train.num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    ##### Save model #####
    torch.save(model.state_dict(), save_model / 'model.pth')
    print(f"Saved PyTorch Model State to {save_model / 'model.pth'}")
    cfg.train.checkpoint = save_model / 'model.pth'
    ##### Save at the end ##### 
    OmegaConf.save(config=cfg, f=cfg.paths.log_dir / 'run_config.yaml')
    print("Done!")

    if cfg.datsaset['generate_report']:
        RepoPath = Path('./').resolve()
        Wiki_Path = RepoPath.parent / 'computing_tutorial.wiki'

        plot_figures(cfg,cfg.paths.fig_dir.absolute(),save_figs=True)
        generate_report(Wiki_Path,cfg.paths.fig_dir,cfg,cfg.version)
    
    
if __name__ == '__main__':
    main()