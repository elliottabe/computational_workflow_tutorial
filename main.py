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

def arg_parser(jupyter=False):
    parser = argparse.ArgumentParser(description=__doc__)
    ##### Directory Parameters #####
    parser.add_argument('--save_dir',           type=str, default='./runs/')
    parser.add_argument('--dataset_type',       type=str, default='FashionMNIST')
    ##### Simulation Parameters ##### 
    parser.add_argument('--gpu',                type=int, default= 0)
    parser.add_argument('--version',            type=str, default='testing')
    parser.add_argument('--generate_report',    type=s2b, default=True)
    ##### Model Paremeters #####       
    
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    
    args = arg_parser()
    save_dir = Path(args['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path('./data/{}/raw/'.format(args['dataset_type']))
    exp_dir_name = args['dataset_type']
    ModelID = 'Testing_{}'.format(args['dataset_type'])
    version = args['version']


    exp = Experiment(name='{}'.format(ModelID),
                        save_dir=save_dir / exp_dir_name, 
                        debug=False,
                        version=version)

    save_model = exp.save_dir / exp.name / 'version_{}/media'.format(version)
    fig_path = exp.save_dir / exp.name / 'version_{}/figures'.format(version)
    fig_path.mkdir(parents=True, exist_ok=True)

    config_path = Path('./conf/config.yaml')
    config = OmegaConf.load(config_path)
    config.args = {'dataset_type': args['dataset_type'], 'version': version, 'save_dir': save_dir, }
    print(OmegaConf.to_yaml(config))

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
    train_dataloader = DataLoader(training_data, batch_size=config.model.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.model.batch_size)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=config.model.lr)

    ##### Save config at start #####
    OmegaConf.save(config, save_model.parent / 'config.yaml')

    # Train and test the model
    for t in range(config.train.num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    ##### Save model #####
    torch.save(model.state_dict(), save_model / 'model.pth')
    print(f"Saved PyTorch Model State to {save_model / 'model.pth'}")
    config.train.checkpoint = save_model / 'model.pth'
    OmegaConf.save(config=config, f=save_model.parent / 'config.yaml')
    print("Done!")

    if args['generate_report']:
        RepoPath = Path('./').resolve()
        Wiki_Path = RepoPath.parent / 'computing_tutorial.wiki'

        plot_figures(config,fig_path.absolute(),save_figs=True)
        generate_report(Wiki_Path,fig_path,config,version)