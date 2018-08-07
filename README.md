## About
(In Progress) This is a research project on Deep Generative Models with Feature Adversarial Learning to learn topological signatures on 2-fold manifold.


### Work in progress
We are refactoring the code to prepare for a release of source code. If you are interested in the work, please feel free to follow us on Github.

### Requirements

Please use `Anaconda` to set up a virtual environment for main Python 3.6 packages. If you have not known what is `Anaconda`, please go to the following [link](!https://conda.io/docs/user-guide/install/index.html) for more information. Then type this in a terminal,
```
conda env create -n env_feature_adversarial_with_topo_signatures.yml
```
Note: Here is `environment` is a name I put into for repository. There will be some other installation for third-party packages, I will put instruction later. However, I would recommend to quickly move forward to GUDHI. So this repository is only for temporary use.

#### Install Torch

```
conda install pytorch torchvision -c pytorch
```

#### Install gudhi
```
conda install -c vincentrouvreau gudhi
```

### How to run source
```
python main.py
```
