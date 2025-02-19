# DeepBioisostere
Deep Learning-based Bioisosteric Replacements for Optimization of Multiple Molecular Properties

# Table of Contents
- [Install Dependencies](#install-dependencies)
- [Training data for DeepBioisostere](#training-data-for-deepbioisostere)
- [MMP Analysis](#mmp-analysis)
- [Training DeepBioisostere](#training-deepbioisostere)
- [Optimize a molecule with DeepBioisostere](#optimize-a-molecule-with-deepbioisostere)


## Install Dependencies
DeepBioisostere model requires conda environment. After installing [conda](https://www.anaconda.com/), you can manually install the required packages as follows:

- rdkit=2022.03.1
- matplotlib
- scipy
- numpy
- scikit-learn
- pytorch>=1.11.0

Or simply you can install the required packages by running
```bash
conda env create -f environment.yml
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

This will configure a new conda environment named 'Bioiso'.
## Training data for DeepBioisostere
1. If you want to re-train DeepBioisostere model *without* data generation, you can download the training data with:
(this script would be provided soon...)
```
./download_train_data.sh
```
And go to [Training DeepBioisostere](#training-deepbioisostere).

2. Or, if you want to re-train DeepBioisostere model *with* data generation by MMP analysis, you can download the ingredients with:
(this script would be provided soon...)
```
./download_mmpa_data.sh
```
And go to [MMP Analysis](#mmp-analysis).

## MMP Analysis
All the necessary source code files are in:
```
./data
```

## Training DeepBioisostere
After getting data for training by ```./download_train_data.sh``` or manually running MMPA, you can re-train a new model by:
```
python ./train_main.py
```

Training arguments that were used to train DeepBioisostere model in our paper can be found in `jobscripts/submit_train.sh`.

And go to [Optimize a molecule with DeepBioisostere](#optimize-a-molecule-with-deepbioisostere).

## Optimize a molecule with DeepBioisostere
An example for molecule optimization with DeepBioisostere can be found in `./example.py` and `./example.ipynb` files.
The process can be divided as 1) initializing `DeepBioisotere` model, 2) initializing `Generator` class, and 3) molecule optimization.

For the molecule optimizaiton process, we provide two options about leaving fragment selection; 1) selection by DeepBioisostere model and 2) manual selection. Below are the full descriptions about the overall process and the two options.

```python
from rdkit import Chem
from scripts.conditioning import Conditioner
from scripts.generate import Generator
from scripts.model import DeepBioisostere
from scripts.property import calc_logP, calc_Mw, calc_QED, calc_SAscore


# Setting smiles to optimize
smi1 = "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O"
smi2 = "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1"

# USER SETTINGS
device = "cpu"
num_cores = 4
batch_size = 512
num_sample_each_mol = 100
new_frag_type = "all"      # one of ["test", "train", "valid", "all"]
properties_to_control = ["mw", "logp"]  # You don't need to worry about the order!

# Set model and fragment library paths
properties = sorted(properties_to_control)
model_path = f"/home/share/DATA/mseok/FRAGMOD/trained_models/DeepBioisostere_{'_'.join(properties)}.pt"
frag_lib_path = "/home/share/DATA/mseok/FRAGMOD/240204/"

# Initialize model and generator
model = DeepBioisostere.from_trained_model(model_path, properties=properties)
conditioner = Conditioner(
    phase="generation",
    properties=properties,
)
generator = Generator(
    model=model,
    processed_frag_dir=frag_lib_path,
    conditioner=conditioner,
    device=device,
    num_cores=num_cores,
    batch_size=batch_size,
    new_frag_type=new_frag_type,
    num_sample_each_mol=num_sample_each_mol,
    properties=properties,
)

# Option 1. Generate with DeepBioisostere
print("Option 1. Generate with DeepBioisostere.")
start_time = time.time()
input_list = [
    (smi1, {"mw": 0, "logp": -1}),
    (smi2, {"mw": 0, "logp": -1}),
]
result_df = generator.generate(input_list)
result_df.to_csv("generation_result.csv", index=False)
print("Elapsed time: ", time.time() - start_time)

# Option 2. Generate with a specific leaving fragment
print("Option 2. Generate with a specific leaving fragment.")
start_time = time.time()
input_list = [
    (smi1, "[*]c1ccccc1[*]", 4, {"mw": 0, "logp": -1}),
    (smi2, "[*]c1ccccn1", 12, {"mw": 0, "logp": -1}),
]
result_df = generator.generate_with_leaving_frag(input_list)
result_df.to_csv("generation_result_with_leaving_frag.csv", index=False)
print("Elapsed time: ", time.time() - start_time)
