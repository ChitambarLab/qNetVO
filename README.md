# pennylane-nonlocality-optimization

*Tune quantum networks with variational quantum optimization.*

## Project Goals

1. Develop a variational quantum optimziation framework for quantum networks:
    1. Construct ansatz circuits that simulate quantum networks.
    2. Provide cost functions for quantifying useful network properties.
    3. Tools for simple optimizations over quantum network parameters.
2. Use framework to characterize nonlocality in noisy, near-term quantum networks:
    1. Find network parameters for maximal nonlocality in ideal quantum networks.
    2. Construct accurate noise models for near-term quantum network simulations. 
    3. Evaluate the noise robustness of quantum network nonlocality.

## Install Project Dependencies

Navigate to project root directory and create the `pennylane-nonlocality` environment
from the `enviroment.yml`:

```
conda env create -f environment.yml
```

Activate the created environment:

```
conda activate pennylane-nonlocality
```
