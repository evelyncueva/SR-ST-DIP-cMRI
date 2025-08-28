# Super Resolution for a Slice and Time Dependent Deep Image Prior in cardiac Cine-MR.

Code by Tabita Catalán and Rafael de la Sotta.  
2025 version maintained by **Evelyn Cueva**.

This release includes code for **dynamic MRI reconstruction** using **Slice- and Time-dependent Deep Image Prior (ST-DIP)** and **spatial super-resolution**.  
It combines and adapts the original repositories by Tabita and Rafael so that they work together. Additional comments and clarifications have been added to facilitate usage.

Original repositories:  
- [Tabita Catalán – NF-cMRI](https://github.com/tabitaCatalan/NF-cMRI)  
- [Rafael de la Sotta – st-DIP-SR](https://github.com/rafadelasotta/NF-cMRI/tree/st-DIP)

This library contains code for MRI reconstructions using different **self-supervised deep learning techniques** such as **Deep Image Prior (DIP)**.  

It is built on [JAX](https://jax.readthedocs.io/en/latest/index.html), a machine learning framework with a NumPy-like API and additional tools for automatic differentiation, compilation, batching, and GPU support. The repository also uses [Flax](https://flax.readthedocs.io/en/latest/) for neural networks and [Optax](https://optax.readthedocs.io/en/latest/) for optimization.

---

## Installation

The code can be installed via `pip` as a package named `inrmri`.

**Quick overview:**

1. Create an isolated environment (`conda` or `venv`) to manage dependencies.  
2. Install JAX following the [official installation guide](https://jax.readthedocs.io/en/latest/installation.html).  
3. Clone the `SR-ST-DIP-cMRI` repository.  
4. Install it in editable mode with `pip` to enable `SR-ST-DIP-cMRI/import inrmri`.

---

### 1. Create an isolated environment

Example with **Conda**:

```bash
$ conda create -n jaxenv python=3.11
$ conda activate jaxenv
```

---

### 2. Install JAX

Follow the [official installation instructions](https://jax.readthedocs.io/en/latest/installation.html).  
The easiest option (requires CUDA 12 drivers):

```bash
(jaxenv)$ pip install -U "jax[cuda12]"
```

---

### 3. Clone the `SR-ST-DIP-cMRI` repository

```bash
(jaxenv)$ git clone https://github.com/evelyncueva/SR-ST-DIP-cMRI
```

This will create a `SR-ST-DIP-cMRI` folder inside the current directory. Install it in editable mode:

```bash
(jaxenv)$ cd SR-ST-DIP-cMRI
(jaxenv)$ pip install -e .
```

- `-e` allows you to edit the library code without reinstalling.  
- `pip install -e .` installs all dependencies listed in `setup.py`.  
  (JAX is not listed there, so install it first as shown above.)

---

## Citation

If you use this code in your research, please cite the original authors:  
- Tabita Catalán – NF-cMRI  
- Rafael de la Sotta – st-DIP-SR

---

## License

MIT License (check the `LICENSE` file for details).
