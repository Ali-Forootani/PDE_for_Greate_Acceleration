# GN-SINDy: Greedy Sampling Neural Network in Sparse Identification of Nonlinear Partial Differential Equations

We introduce the Greedy Sampling Neural Network in Sparse Identification of Nonlinear Partial Differential Equations (__GN-SINDy__), a pioneering approach that seamlessly integrates a novel greedy sampling technique, deep neural networks, and advanced sparsity-promoting algorithms. Our method not only addresses the formidable challenges posed by the curse of dimensionality and large datasets in discovering models for nonlinear __PDE__ s but also sets a new standard for efficiency and accuracy by redefining the data collection and minimization units within the __SINDy__ framework. By combining the strengths of these diverse techniques, __GN-SINDy__ represents a leap forward in the realm of model discovery, promising unprecedented insights into the intricate dynamics of complex systems.



![Alt text](paper/tikz_picture/GNSINDy.png)



## Installation
I assume that you already have `linux` OS to create a new `conda` environemnt.
 
To install the packge open a new `bash` terminal in your local directory and type

```bash
conda create --name GNSINDy python=3.10.6
```

This will create a conda environment with the default name `GNSINDy`.



To activate the conda environment type on the `bash` terminal 

```bash
conda activate GNSINDy
```
Then start to install the following packages:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::numpy
conda install conda-forge::scipy
conda install conda-forge::matplotlib
conda install conda-forge::tqdm
conda install conda-forge::pandas
conda install conda-forge::scikit-learn
conda install conda-forge::tensorboard
conda install conda-forge::natsort
conda install conda-forge::spyder
```



If you `do not want to use conda` you can create a virtual environment as explained in the following.


```bash
python -m venv <env_name>
virtualenv <env_name>
```

Activate the virtual environment based on your operating system:

• On Windows:

```bash 
.\<env_name>\Scripts\activate
```

• On linux or MacOS:

```bash 
source <env_name>/bin/activate
```


A `Spyder` IDE is already installed together with the environment. When your conda environment is activated type `Spyder` so you can use it.

__hint__: the root folder should be named `GNSINDy` so you can execute the modules. In case you download the repository and extract it in your local machine you have to consider this! Nevertheless, the root directory is `GNSINDy` and different modules are imported e.g. `from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader`

If you `do not want to use conda` you can create a virtual environment as explained in the following.

```bash
python -m venv <env_name>
```

Activate the virtual environment based on your operating system:

• On Windows:

```bash 
.\<env_name>\Scripts\activate
```

• On linux or MacOS:

```bash 
source <env_name>/bin/activate
```
 
After creating a new virtual environment by using `python -m venv GNSINDy` then activate it as mentioned above then install the packages as follows

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy==1.23.5
pip install scipy==1.11.4
pip install matplotlib==3.5.3
pip install tqdm==4.65.0
pip install fastapi==0.96.1
pip install scikit-learn==1.0.2
pip install pandas pandas==1.5.2
pip install --upgrade transformers
pip install tensorflow natsort
```



## Usage

In the __/src/examples__ folder you can find different simulations. We considered __fertilizer_stat.py__ as the first example. In the __src/data__ folder you will find the dataset that are used in the article. In the __deepymod__ folder you can find different modules that are used in the package.


## Contributing

Feel free to clone the repository and extend.


## Where to find us?
Max Planck Institute of Geoanthropology, Jena, Germany.
Max Planck Institute for Dynamics of Compelx Technical Systems, CSC group, Magdeburg, 39106, Germany.
You can either drop an email to the authors.

Email Me: (forootani@gea.mpg.de/forootani@mpi-magdeburg.mpg.de/alifoootani@ieee.org)




## License

[MIT](https://choosealicense.com/licenses/mit/)
