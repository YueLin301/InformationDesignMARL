## Information Design in Multi-Agent Reinforcement Learning

This repository presents a hasty and rudimentary implementation 
of all experiments mentioned in [our paper (arxiv version)](). 
In this version, the constrained optimization problem is implemented by dual gradient descent method.

All the experimental results can be reproduced directly through this repo, 
if you configure your own [wandb](https://wandb.ai) keys correctly.

It's important to mention that for each experiment, the algorithm is written separately,
rather than creating a separate class that's independent of the experiments. 
This is for the **decoupling** between experiments, 
so as to facilitate rapid iteration and trial-and-error.


## For Reproduction
You can run the experiments by running `main.py` in the corresponding `exp_` prefix folder.
1. Create a file `mykey.py` in the `exp_` prefix folder. 
Experiments that have been successful and have relatively good performance include:
    - Recommendation
    - Reaching Goals (map `3x3`)
2. Edit `mykey.py`:
```python
wandb_login_key = '' # Your login API key
wandb_project_name = ''
wandb_entity_name = ''
```
3. Check your wandb configuration and fill the empty strings with your private information. 
See the [wandb quickstart](https://docs.wandb.ai/quickstart).
4. If you have any queries or concerns, please do not hesitate to reach out to me. 