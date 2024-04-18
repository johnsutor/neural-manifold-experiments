# Manifold Experiments 
Experiments regarding settings for MMCR loss for video data. This repo uses [Hydra](https://hydra.cc/) for configuration. To alter default values (as printed at the beginning of training), please refer to the [Override Syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) 

## Running 
Make sure to install all required packages first using
```shell
$ pip install -r requirements.txt
```
Then, make sure to set your experiment root directory like so
```shell
$ export EXPERIMENT_ROOT=<YOUR EXPERIMENT ROOT DIRECTORY>
```
Next, make sure to install the package to enable running it on SLURM
```shell
$ pip install -e . 
```
Finally, use 
```shell
$ python train.py
```
passing any overrides based on the above link to Hydra. To use with the submitit launcher, please 
reference the instructions [here](https://hydra.cc/docs/1.3/plugins/submitit_launcher/)