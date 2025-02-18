

## Install CALVIN

```bash
conda create -n calvin python=3.8 
pip install --upgrade pip setuptools==57.5.0 
```
Then run the `sh install.sh` script from calvin repo.

The install:
```bash
pip install -U diffusers accelerate
pip install --upgrade torch torchvision  # torch 2.4.1
```