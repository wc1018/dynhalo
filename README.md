# Dynamics-Based Halo Model

This code reproduces Salazar et. al. (2024) (arxiv:XXXX).

## Requirements
Setting up the environment turned up to be a bit tricky with so many dependencies. The  first thing is to do is install `corrfunc` and make sure multiprocessing is working properly. Then install `NBodyKit` using conda. Finally, you can pip install this repo.

### [`Corrfunc`](https://github.com/manodeep/Corrfunc)
To compute correlation functions with multithreading, you'll need to install `corrfunc` from source.
```sh
$ conda create --name <env name> python=3.8 numpy cython mpi4py gcc gsl
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
$ conda activate <env name>
$ python -m pip install .
```
I do recommend you run the tests as suggested in the [`corrfunc` package](https://github.com/manodeep/Corrfunc?tab=readme-ov-file#method-1-source-installation-recommended).

If you get an error where `crypt.h` could not be found, simply copy it into your environment.
```sh
$ cp /usr/include/crypt.h /home/<user>/miniconda/env/<env name>/include/python3.8/
```

### [`NBodyKit`](https://github.com/bccp/nbodykit)
<!-- There is two dependencies, `kdcount` and `classylss`, that have a hard time compiling (in my experience). If you encounter this issue, try installing them first
```sh
$ conda activate <env_name>
$ conda install -c bccp kdcount classylss
``` -->

It can be done from souce via conda. Follow the installation instructions in the documentation to install `nbodykit`
```sh
$ conda activate dhmenv
$ conda install -c bccp nbodykit
```

`Nbodykit` installs its own compiled version of `corrfunc`. However, this should not create a conflict or override the previous installation. If you find that multithreading is not working properly after installing `NBodyKit` try removing `corrfunc` and reinstalling the one you cloned before:
```sh
$ conda activate <env_name>
$ pip uninstall corrfunc
$ cd Corrfunc
$ python -m pip install .
```

To update:
```sh
$ conda update -c bccp --all
```

## Installation
Install this package via
```sh
$ git clone https://github.com/edgarmsalazar/Dynamics-Based-Halo-Model
$ pip install .
```