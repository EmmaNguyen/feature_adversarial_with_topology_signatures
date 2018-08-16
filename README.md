## About
(In Progress) This is a research project on Deep Generative Models with Feature Adversarial Learning to learn topological signatures on 2-fold manifold.


### Work in progress
We are refactoring the code to prepare for a release of source code. If you are interested in the work, please feel free to follow us on Github.

### Requirements

Please use `Anaconda` to set up a virtual environment for main Python 3.6 packages. If you have not known what is `Anaconda`, please go to the following [link](!https://conda.io/docs/user-guide/install/index.html) for more information. Then type this in a terminal,
```
conda create -n <virtual_env_name> python=3.6
```
To install all packages, please continue with following command,
```
pip install -r requirements.txt
```
### Run a test on setup

```
python -m unittest
```

### How to run source
Below is the main file to trigger all important settings along with experiments.
Expected result will be at `../data/`

```
python main.py --no-cuda
```

#### Description
```
usage: main.py [-h] [--batch-size N] [--epochs N] [--lr LR] [--alpha A]
               [--distribution DIST] [--no-cuda] [--num_workers N] [--seed S]
               [--log-interval N]

PyTorch Implementation

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 500)
  --epochs N           number of epochs to train (default: 30)
  --lr LR              learning rate (default: 0.001)
  --alpha A            RMSprop alpha/rho (default: 0.9)
  --distribution DIST  Latent Distribution (default: circle)
  --no-cuda            disables CUDA training
  --num_workers N      number of dataloader workers if device is CPU (default:
                       8)
  --seed S             random seed (default: 7)
  --log-interval N     number of batches to log training status (default: 10)
```
