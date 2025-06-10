# [WIP] LangToMo


## Installation

```
Python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U diffusers accelerate transformers
pip install einops matplotlib wandb
pip install dm-reverb[tensorflow] tensorflow-datasets rlds
pip install "pydantic>=2.0" --upgrade
```


## Sythetic Setup - CALVIN
This section explains using CALVIN dataset to train our model.


### Data Generation
To generate optical flow and save, run following script:
```bash
cd src/dataset
python generate_flow.py
```

To visualize generation data, run
```bash
python test/dataset/test_generated_flow.py
```

Additional code for visualization and sanity checks on CALVIN optical flow generation
are found in `src/dataset/calvin.py`.


## Training
```bash
python src/train.py --output-dir test_0XX
```
