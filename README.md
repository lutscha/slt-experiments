### Edge of Stability Normalization Experiments

This repo contains code for investigating the interaction between the [Edge of Stability](https://arxiv.org/abs/2103.00065) and weight decay. The implementation is based on
that which is provided in the EoS paper ([implementaion found here](https://github.com/locuslab/edge-of-stability)).


## Example usage
Set the enviroment variables
`export DATASET="/my/directory/datasets"`
`export RESULTS="/my/directory/results`


### Run in script
```python
from gd import main

main(
    dataset="cifar10-5k",
    arch_id='full_200', 
    loss="mse",
    opt="gd",
    lr=0.02,
    wd=0.02,
    max_steps=14000,
    eig_freq=25,
    neigs=1,
    iterate_freq=-1,
    save_freq=5000,
    save_model=True,
    seed=1,
)
```
The training metrics,$\lambda_{max}$, $\alpha$ and $c_y$ will be saved as tensors in the `RESULTS` path

Architectures used in paper
- CNN: arch_id="cnn_relu"
- MLP: arch_id="full_200"
