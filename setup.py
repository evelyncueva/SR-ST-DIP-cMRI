from setuptools import setup

setup(
    name='inrmri',
    version='0.3.0',
    packages=['inrmri'],
    author='Tabita Catalan',
    author_email='tabicm.nhg@gmail.com',
    install_requires=[ # supongo que Jax ya está instalado, no lo pongo aquí porque suele ser un cacho de instalar 
        'flax',
        'optax',
        'image-similarity-measures',
        'ipykernel',
        'matplotlib',
        'numpy',
        'scikit-image',
        'scipy',
        'tqdm',
        'wonderwords',
    ]
)