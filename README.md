# QNetOptimizer

*A Python library for tuning quantum networks with variational quantum optimization.*

## Project Goals

1. Develop a variational quantum optimziation framework for quantum networks:
    1. Construct ansatz circuits that simulate quantum networks.
    2. Provide cost functions for quantifying useful network properties.
    3. Tools for simple optimizations over quantum network parameters.
2. Use framework to characterize nonlocality in noisy, near-term quantum networks:
    1. Find network parameters for maximal nonlocality in ideal quantum networks.
    2. Construct accurate noise models for near-term quantum network simulations. 
    3. Evaluate the noise robustness of quantum network nonlocality.

## Development

### Environment

For convenience a dev environment can be setup from the `environment.yml`.

1. Create the `pennylane-nonlocality-dev` enviroment:

```
(base) $ conda env create -f environment.yml
```

2. Activate the `pennylane-nonlocality-dev` environment:

```
(base) $ conda activate pennylane-nonlocality-docs
```

### Documentation

To view the project documentation it must be built and served locally. 
It cannot currently be hosted by GitHub because the repository is private and hosting
services are not available for free accounts.


1. Build documentation:

```
(pennylane-nonlocality-docs) $ sphinx-build -b html docs/source/ docs/build/html
```

2. Locally serve documentation by navigating to the `./docs/build/html` directory and running:

```
(pennylane-nonlocality-docs) $ python -m http.server --bind localhost
``` 

### Tests

Run tests from the root directory with with `$ python -m pytest`. All test are found in the `./test` directory.

### Code Formatting

To format code run `$ black -l 100 QNetOptimizer test docs`. This uses the [black](https://black.readthedocs.io/en/stable/) code formatter.


