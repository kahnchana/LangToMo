# LangToMo


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