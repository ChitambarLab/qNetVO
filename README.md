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

## Installing Project Dependencies

Navigate to project root directory and create the `pennylane-nonlocality` environment
from the `enviroment.yml`:

```
conda env create -f environment.yml
```

Activate the created environment:

```
conda activate pennylane-nonlocality
```

## Documentation

The project documentation must be built and viewed. 
It cannot currently be hosted by GitHub because the repository is private and hosting
services are not available for free accounts. 

1. Create the `pennylane-nonlocality-docs` conda enviroment:

```
(base) $ conda env create -f docs/environment.yml
```

2. Activate the `pennylane-nonlocality-docs` conda environment:

```
(base) $ conda activate pennylane-nonlocality-docs
```

3. Build documentation:

```
(pennylane-nonlocality-docs) $ sphinx-build -b html docs/source/ docs/build/html
```

4. Locally serve documentation by navigating to the `./docs/build/html` directory and running:

```
(pennylane-nonlocality-docs) $ python -m http.server --bind localhost
``` 

## Development

### Tests

Run tests from the root directory with with `$ python -m pytest`. All test are found in the `./test` directory.

### Code Formatting

To format code run `$ black -l 100 QNetOptimizer test`. This requires the [black](https://black.readthedocs.io/en/stable/) code formatter which can be installed with `$ pip install black`.


