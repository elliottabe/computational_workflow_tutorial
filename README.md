# computing_tutorial

Slide available at: [BruntonLab Meeting 08/24/23](https://docs.google.com/presentation/d/1Hz9PwWblYLvo1TFiZbOrGMarjNNcP17WNFkGSrxVUgM/edit?usp=sharing)

# Typical setup for installing conda environment and dependencies

To install the repo there is a conda environment that will install the necessary packages. Make sure you are in the Github repo directory.
Use command:
conda env create -f environment.yaml

After installing activate the conda environment:
conda activate comp_workflow

Once in the environment go to this site to install the appropriate pytorch version. I would recommend using conda:
https://pytorch.org/get-started/locally/

After pytorch is correctly installed run this command to install pip reqruirements:
pip install -r requirements.txt