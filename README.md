# qNetVO: The Quantum Network Variational Optimizer

*Simulate and optimize quantum communication networks using quantum computers.*

[![Test Status](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml)

## Features

QNetVO simulates quantum communication networks on differentiable quantum cicuits.
The cicuit parameters are optimized with respect to a cost function using gradient descent.
qNetVO is powered by `PennyLane <https://pennylane.ai>`_ an open-source framework
for cross-platform quantum machine learning.

### Simulating Quantum Communication Networks:

* Construct complex quantum network ansatzes from generic quantum circuit compenents.
* Simulate the quantum network on a quantum computer or classical simulator.

### Optimizing Quantum Communication Networks:

* Use our library of network oriented cost functions or create your own.
* Gradient descent methods for tuning quantum network ansatz settings to minimize the cost.

## Quick Start

Install qNetVO:

```
$ pip install qnetvo
```

Install PennyLane:

```
$ pip install pennylane==0.20
```

Import packages:

```
import pennylane as qml
import qnetvo as qnet
```

## Development

### Creating the `qnetvo-dev` Environment

The [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary) distribution of Python is used to ensure a consistent development environment.
Follow the Anaconda [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) to setup the `conda` command line tool for your
operating system.
The `conda` tool creates the dev environment from the `environment.yml` file.
For more details on how to use `conda` see the [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) page in the `conda` documentation.

To create the dev environment, navigate to the root directory of the `qNetVO/` repository and follow these steps.

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

All packages for building docs, running tests, and running notebooks are installed.
Changes to the local `./src/qnetvo` codebase are reflected whenever `import qnetvo` is called.

### Running Tests

Tests are found in the `./test` directory and run using [`pytest`](https://docs.pytest.org/en/7.0.x/).
First, setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment).
Then, from the root directory, run:

```
(qnetvo-dev) $ pytest
```

### Running Demos

Demos are found in the `./demos` directory and implemented in Jupyter notebooks.
To run the demos locally, first setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment), then run:

```
(qnetvo-dev) $  jupyter-notebook
```
A Jupyter notebook server will launch in your browser allowing you to run the notebooks.

### Building Documentation

It is important to view the documentation before committing changes.
To locally build and view the documentation, first setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment).
Then, follow these steps.


1. **Build Documentation:** From the root directory run:

```
(qnetvo-dev) $ sphinx-build -b html docs/source/ docs/build/html
```

2. **Serve Documentation:** Navigate to the `./docs/build/html` directory and run:

```
(qnetvo-dev) $ python -m http.server --bind localhost
```

3. **View Documentation:** Copy and paste the returned IP address to your browser.

### Formatting Code

All code in this project is autoformatted using [black](https://black.readthedocs.io/en/stable/).
First, setup the [dev environment](https://github.com/ChitambarLab/qNetVO#environment).
Then, from the root directory, run:

```
(qnetvo-dev) $ black -l 100 src test docs
```


