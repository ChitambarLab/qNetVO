Contributing
============

We welcome outside contributions to qNetVO.
For small changes, please fork the repository and make a pull request containing you change.
For large contributions, please contact the corresponding author at :email:`brian.d.doolittle@gmail.com`.

All contributed code must have passing tests, comprehensive documentation,
and proper code formatting.
Development instructions are found below.

Development Environment
-----------------------

For convenience and reproducibility, contributors should use the ``qnetvo-dev`` conda environment.
The `Anaconda <https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary>`_
distribution of Python ensures a consistent development environment.
Follow the Anaconda `installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation>`_ to set up the ``conda`` command line tool for your
operating system.
The ``conda`` tool creates the dev environment from the ``environment.yml`` file.
For more details on how to use ``conda`` see the `managing environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ page in the ``conda`` documentation.

To create the dev environment, navigate to the root directory of the ``qNetVO`` repository and follow these steps.

1. Create the ``qnetvo-dev`` conda environment:

.. code-block::

    (base) $ conda env create -f environment.yml

2. Activate the ``qnetvo-dev`` conda environment:

.. code-block::

    (base) $ conda activate qnetvo-dev

3. Install the local ``qnetvo`` package in editable mode:

.. code-block::

    (qnetvo-dev) $ pip install -e .

At this point, all packages for running tests, building docs, and running notebooks are installed.
Changes to the local ``./src/qnetvo`` codebase will be reflected when ``import qnetvo`` is called in Python code.

Running Tests
-------------

Tests are found in the ``./test`` directory and run using |pytest|_
First, set up the `Development Environment`_.
Then, from the root directory, run:

.. code-block::

    (qnetvo-dev) $ pytest

.. |pytest| replace:: ``pytest``
.. _pytest: https://docs.pytest.org/en/7.0.x/

Running Demos
-------------

Demos are found in the `./demos` directory and implemented in Jupyter notebooks.
To run the demos locally, first set up the `Development Environment`_, then run:

.. code-block::

    (qnetvo-dev) $  jupyter-notebook

A Jupyter notebook server will launch in your browser allowing you to run the notebooks.

Building Documentation
----------------------

It is important to view the documentation before committing changes.
To locally build and view the documentation, first set up the `Development Environment`_.
Then, follow these steps.

1. **Build Documentation:** From the root directory run:

.. code-block::

    (qnetvo-dev) $ sphinx-build -b html docs/source/ docs/build/html

2. **Serve Documentation:** Navigate to the ``./docs/build/html`` directory and run:

.. code-block::

    (qnetvo-dev) $ python -m http.server --bind localhost

3. **View Documentation:** Copy and paste the returned IP address to your browser.

Formatting Code
---------------

All code in this project is autoformatted using `black <https://black.readthedocs.io/en/stable/>`_.
First, set up the `Development Environment`_.
Then, from the root directory, run:

.. code-block::

    (qnetvo-dev) $ black -l 100 src test docs

Semantic Versioning
-------------------

This project uses `semantic versioning <https://semver.org/>`_ to manage
releases and maintain consistent software.

Packaging and Releases
----------------------

This project is packaged using PyPI, the `Python Package Index <https://pypi.org/>`_.
Please refer to `this tutorial <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_ for details on releasing a new version.
