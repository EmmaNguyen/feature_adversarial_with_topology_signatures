# feature_adversarial_with_topology_signatures
Research on Deep Generative Models with Feature Adversarial Learning to learn topological signatures on 2-fold manifold


# How to install

### Install Anaconda and virtual environment

### Install torch 0.3

```
mkl=2018 pytorch=0.3.0 -c pytorch -c intel
```

#### Binary file distributed by Anaconda
```
conda install pytorch torchvision -c pytorch
```

If running GPUs, please check that you have a appropriate set up as mine to prevent any headache troublesome by running this unittest.
```
python unittests/test_torch_setting.py
```
#### Source file from github

```
git clone

```

Probably, other steps that you would need to do to make the code run.

### Install other Python packages

```
conda install -n name_of_env --file dev_py35
```

### Install Computational packages for topological signatures

```
git clone https://github.com/DIPHA/dipha
```

```
cmake CMakeLists.txt
make
cd ../dipha && cmake dipha
sudo chmod +x dipha
```

Try with `dipha`

```bash
gcc dipha
```

### How to run a file
```
python refactor_nips_chris_et_al.py --data_path=/home/emma/Research/GAN/nips2017/data/dgm_provider/npht_small_train_32dirs.h5
```
