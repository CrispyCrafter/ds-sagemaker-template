# Ramp Sagemaker CookieCutter

## Usage
This repo acts as centralised starting point for all Sagemaker Models at ramp. 

To use run the following:

```shell
pip install cookiecutter==2.3.0
cookiecutter gh:rampgrowth/ds-sagemaker-template
```

This will generate the following package structure on your local machine:
```
<project_name>
├── README.md
├── main.py
├── src
│   ├── __init__.py
│   ├── base.py
│   ├── data.py
│   ├── loss.py
│   └── model.py
├── requirements.txt
├── requirements_dev.txt
└── config.json
```
## Overview
The following modules are shipped with the base cookiecutter template.
All files can be modfied based on user prefference or updated to implementation specific requirements. 

*NB*: __However__, at the very miniumum the following modules need to be present:

```
<project_name>
├── src
│   ├── base.py
│   ├── data.py
│   ├── loss.py
│   └── model.py
├── main.py
└── requirements.txt
```

### `train.py`
This is the main entypoint used by sagemaker to train your defined model

### `models.py` and `base.py`
Custom torch models need to defined in this module. All models need to inherit from `base.BaseNet` which impliments the following:

1. Pytorch Categorical Embeddings
1. training and validation steps
1. Weights and Biases logging
1. Optimisers
1. ONNX: 
    * `example_input_array` -> Used to generate the ONNX model 
    * `export_model` -> Model export definitions

 ### `data.py`
 Dataloader and data-prep defintions

 ### `features.py`
Datatransformation operations. Please bear in mind all modules imported in this folder, need to exist on the inference machine at production runtime.

For instance the base template makes use of:
```python
import pandas as pd
import numpy as np
import sklearn
```

This implies that a target machine (`engine/runway`) will have these modules installed. Please ensure this is the case.

Relative imports are strictly not allowed. For instance the following operation in illegal:
```python
from .utils import some_util_function
from src.utils import some_util_function
```