# qNetVO: Quantum Network Variational Optimizer

*Simulate and optimize quantum communication networks using quantum computers.*

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chitambarlab.github.io/qNetVO/index.html)[![Test Status](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/qNetVO/actions/workflows/run_tests.yml)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![PyPI version](https://badge.fury.io/py/qNetVO.svg)](https://badge.fury.io/py/qNetVO)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6345834.svg)](https://doi.org/10.5281/zenodo.6345834)

## Features

QNetVO simulates quantum communication networks on differentiable quantum cicuits.
The cicuit parameters are optimized with respect to a cost function using automatic differentiation and gradient descent.
QNetVO is powered by [PennyLane](https://pennylane.ai), an open-source framework
for cross-platform quantum machine learning.

### Simulating Quantum Communication Networks:

* Construct complex quantum network ansatzes from generic quantum circuit compenents.
* Simulate the quantum network on a quantum computer or classical simulator.

### Optimizing Quantum Communication Networks:

* Use our library of network-oriented cost functions or create your own.
* Gradient descent methods for tuning quantum network ansatz settings to minimize the cost.

## Quick Start

Install qNetVO:

```
$ pip install qnetvo
```

Install PennyLane:

```
$ pip install pennylane==0.37
```

Import packages:

```
import pennylane as qml
import qnetvo as qnet
```

<div class="admonition note">
<p class="admonition-title">
Note
</p>
<p>
For optimal use, qNetVO should be used with PennyLane.
QNetVO is currently compatible with PennyLane v0.37.
</p>
</div>

## Contributing

We welcome outside contributions to qNetVO.
Please see the [Contributing](https://chitambarlab.github.io/qNetVO/contributing.html)
page for details and a development guide. 

## How to Cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6345835.svg)](https://doi.org/10.5281/zenodo.6345835)

See [CITATION.bib](https://github.com/ChitambarLab/qNetVO/blob/main/CITATION.bib) for a BibTex reference to qNetVO.

## License

QNetVO is free and open-source.
The software is released under the Apache License, Version 2.0.
See [LICENSE](https://github.com/ChitambarLab/qNetVO/blob/main/LICENSE) for details and
[NOTICE](https://github.com/ChitambarLab/qNetVO/blob/main/NOTICE.txt) for copyright information.

## Acknowledgments

We thank [Xanadu](https://www.xanadu.ai/), the
[UIUC Physics Department](https://physics.illinois.edu/), and the
[Quantum Information Science and Engineering Network (QISE-Net)](https://qisenet.uchicago.edu/)
for their support of qNetVO.
Work funded by NSF award DMR-1747426.
