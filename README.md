# Dynamics-Based Halo Model

This code reproduces Salazar et. al. (2024) (arxiv:XXXX).

## Requirements

### [`Corrfunc`](https://github.com/manodeep/Corrfunc)
To compute correlation functions with multithreading, you'll need to install `corrfunc` from source.
```sh
$ conda create --name dhmenv python=3.8 numpy cython mpi4py gcc gsl
```
You can follow the installation process specified in the `corrfunc` repo to run tests and make sure everything is working as intended. If you wish to simply install the package do the following in another directory:
```sh
$ git clone https://github.com/manodeep/Corrfunc.git
$ cd Corrfunc
$ make
$ make install
```
Then install the package.
```sh
$ conda activate dhmenv
$ python -m pip install .
```
### [`NBodyKit`](https://github.com/bccp/nbodykit)
It can be done from souce via conda. Follow the installation instructions in the documentation to install `nbodykit`
```sh
$ conda activate dhmenv
$ conda install -c bccp nbodykit
```

`Nbodykit` installs its own compiled version of `corrfunc`. However, this should not create a conflict or override the previous installation. If you find that multithreading is not working properly after installing `NBodyKit`:
```sh
$ conda activate dhmenv
$ conda remove corrfunc
$ cd Corrfunc
$ python -m pip install .
```

To update:
```sh
$ conda update -c bccp --all
```

## Installation
We can now install our package via
```sh
$ pip install git+https://github.com/edgarmsalazar/Dynamics-Based-Halo-Model
```