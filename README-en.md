# st-dip-sr

Code by Tabita Catalán and Rafael de la Sotta.  
2025 version maintained by **Evelyn Cueva**.

This release includes code for **dynamic MRI reconstruction** using **Slice- and Time-dependent Deep Image Prior (ST-DIP)** and **spatial super-resolution**.  
It combines and adapts the original repositories by Tabita and Rafael so that they work together. Additional comments and clarifications have been added to facilitate usage.

Original repositories:  
- [Tabita Catalán – NF-cMRI](https://github.com/tabitaCatalan/NF-cMRI)  
- [Rafael de la Sotta – st-DIP-SR](https://github.com/rafadelasotta/NF-cMRI/tree/st-DIP)

This library contains code for MRI reconstructions using different **self-supervised deep learning techniques** such as **Implicit Neural Representations (INRs)** and **Deep Image Prior (DIP)**.  

It is built on [JAX](https://jax.readthedocs.io/en/latest/index.html), a machine learning framework with a NumPy-like API and additional tools for automatic differentiation, compilation, batching, and GPU support. The repository also uses [Flax](https://flax.readthedocs.io/en/latest/) for neural networks and [Optax](https://optax.readthedocs.io/en/latest/) for optimization.

---

## Installation

The code can be installed via `pip` as a package named `inrmri`.

**Quick overview:**

1. Create an isolated environment (`conda` or `venv`) to manage dependencies.  
2. Install JAX following the [official installation guide](https://jax.readthedocs.io/en/latest/installation.html).  
3. Clone the `ST-SR-cMRI` repository.  
4. Install it in editable mode with `pip` to enable `import inrmri`.

---

### 1. Create an isolated environment

Example with **Conda**:

```bash
$ conda create -n jaxenv python=3.11
$ conda activate jaxenv
```

⚠️ Mixing `conda` and `pip` can sometimes cause conflicts (e.g., Flax version mismatches).

Example with **virtualenv**:

```bash
$ python -m venv jaxenv
$ source jaxenv/bin/activate
```

Verify that you are using the environment’s `pip`:

```bash
(jaxenv)$ type pip
pip is /path-to-folder/jaxenv/bin/pip
```

---

### 2. Install JAX

Follow the [official installation instructions](https://jax.readthedocs.io/en/latest/installation.html).  
The easiest option (requires CUDA 12 drivers):

```bash
(jaxenv)$ pip install -U "jax[cuda12]"
```

---

### 3. Clone the `ST-SR-cMRI` repository as a Git submodule

```bash
(jaxenv)$ git submodule add https://github.com/tabitaCatalan/NF-cMRI
```

This will create a `NF-cMRI` folder inside the current directory. Install it in editable mode:

```bash
(jaxenv)$ cd NF-cMRI
(jaxenv)$ pip install -e .
```

- `-e` allows you to edit the library code without reinstalling.  
- `pip install -e .` installs all dependencies listed in `setup.py`.  
  (JAX is not listed there, so install it first as shown above.)

---

### 4. Using Jupyter Notebooks

Instead of installing Jupyter in every environment, you can keep a global installation and just add kernels.  

For environments where you want to run notebooks, install `ipykernel` (already listed in `install_requires`). Then register the kernel:

```bash
python -m ipykernel install --user --name jaxenv
```

---

## Prerequisites

### Install BART

Running the examples requires the [BART Toolbox](https://mrirecon.github.io/bart/), an open-source software package for MRI reconstruction, coil sensitivity estimation, FFT/NUFFT, and more.

You will need BART **in two ways**:

1. **System-wide installation** (so that `bart` is recognized as a shell command).  
   Example:  
   ```bash
   sudo apt install bart
   ```

2. **Python interface available** (so that `import cfl` works).  
   Clone the [BART repository](https://github.com/mrirecon/bart/tree/master) and set the path in `inrmri/bart.py`:

   ```python
   import os, sys

   os.environ["TOOLBOX_PATH"] = "/path/to/bart"  # Replace with your BART repo path
   sys.path.append(os.path.join(os.environ["TOOLBOX_PATH"], "python"))

   import cfl
   ```

---

### Data directories

Some notebooks use the dataset:  
[Replication Data for: Multi-Domain Convolutional Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac MRI](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FCI3WB6&version=2.0).

It is loaded via [`inrmri/data_harvard.py`](inrmri/data_harvard.py). Data is downloaded automatically and stored in the folder specified by `HARVARD_FOLDER`.  
👉 You **must update this variable** to an absolute path of your choice.

Also update the variable `BART_DIRECTORY` to specify where BART-generated files will be stored. Again, use an absolute path.

> Note: When using BART through its Python interface, temporary files are deleted automatically.  
> In this repository, we choose to keep them for reuse in multiple steps.

---

## Citation

If you use this code in your research, please cite the original authors:  
- Tabita Catalán – NF-cMRI  
- Rafael de la Sotta – st-DIP-SR

---

## License

MIT License (check the `LICENSE` file for details).
