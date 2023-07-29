# Play chess FHE

This repository is a tutorial dedicated to create a ML agent that can play chess in FHE using the Concrete-ML library. This work also [compete for a bounty program](https://github.com/zama-ai/bounty-program/blob/main/Bounties/Machine_Learning/create-an-app-play-chess-in-fhe.md).

The application of Fully Homomorphic Encryption (FHE) in deep learning lead us to rethink our process. Indeed, it allows us to encrypt the entire process while preserving the privacy of our data.

In this notebook, we will present a Deep Learning model for playing chess that leverages FHE. The FHE process involves several mandatory steps that significantly impact and change how we traditionally approach deep learning.


This project is designed to participate in the bounty provided by Zama
> https://github.com/zama-ai/bounty-program/issues/32


## Tutorial Notebook

The proposed solution can be found in the `tutorial.ipynb` notebook. To run it, we suggest you to use poetry. Poetry is a tool for dependency in python. You can install it: 

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, you can install python dependencies for this project with:

```bash
poetry install
```

Finally, you can run a notebook with the python dependencies with:

```bash
poetry run jupyter notebook
```

