# qNetVO

*The **Q**uantum **Net**work **V**ariational **O**ptimizer is python framework for
simulating and optimizing quantum communication networks using
quantum hardware.*

[![Test Status](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml)

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

The [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary) distribution of Python is used to ensure a consistent development environment.
Follow the Anaconda [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) to setup the `conda` command line tool for your
operating system.
The `conda` tool is used to create the dev environment from the `environment.yml` file.
For more details on how to use `conda` see the [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) page in the `conda` documentation.

To create the dev environment, navigate to the root `qNetVO/` project directory and follow these steps.

1. Create the `qnetvo-dev` conda environment:

```
(base) $ conda env create -f environment.yml
```

2. Activate the `qnetvo-dev` conda environment:

```
(base) $ conda activate qnetvo-dev
```

3. Install the local `qnetvo` package in editable mode:

```
(qnetvo-dev) $ pip install -e .
```

At this point all packages for building docs, running tests, and running notebooks will be installed.
Local changes made to the `./src/qnetvo/` codebase will be reflected whenever the `import qnetvo` is called in Python code.

### Tests

Tests are run using [`pytest`](https://docs.pytest.org/en/7.0.x/) and located in the `./test/` directory.
To run tests, first setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment), then run:

```
(qnetvo-dev) $ pytest
```

### Documentation

It is important to view the documentation before committing changes to the `qNetVO/` codebase.
To locally build and serve the documentation, first setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment), then follow these steps.


1. **Build Documentation:** From the root directory run:

```
(qnetvo-dev) $ sphinx-build -b html docs/source/ docs/build/html
```

2. **Serve Documentation:** Navigate to the `./docs/build/html` directory and run:

```
(qnetvo-dev) $ python -m http.server --bind localhost
```

3. **View Documentation:** Copy and paste the returned IP address to your browser URL to view the `qnetvo` documentation.

### Code Formatting

All code in this project is autoformatted using the [black](https://black.readthedocs.io/en/stable/) code formatter.
Code must be formatted before a pull request is made.
First, setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment)).
Then, from the root directory run:

```
(qnetvo-dev) $ black -l 100 src test docs
```


