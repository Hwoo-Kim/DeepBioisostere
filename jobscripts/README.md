# Training argument setting

Example script to train DeepBioisostere is provided in `submit_train.sh`, a bash script based on SLURM job scheduler.
For experiments, user is expected to set following arguments:

`raw_data_path`: the absolute path to the training data file (.csv).

`scratch_dir`: the scratch directory to which training data file would be copied. 

`properties`: list of propertes that the DeepBioisostere model is trained to conrol. If multiple properties are used for control target, they must be splitted by comma (,) e.g. `logp,qed`.

`conditioning`: whether to train conditional DeepBioisostere model or not. If `False`, `properties` variable is not used and unconditional DeepBioisostere model is trained.

`save_name`: optional. Default save name is already provided, but if you want to change it, you can manually set this variable as you want.
