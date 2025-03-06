![PythonVersion](https://img.shields.io/badge/python-3.8-succes)
[![License https://github.com/trainindata/feature-engineering-for-time-series-forecasting/blob/master/LICENSE](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/trainindata/feature-engineering-for-time-series-forecasting/blob/master/LICENSE)

# Quantum transport simulations of twisted bilayer graphene for the study of twist angle disorder

Published March 2025

## Research goals

In this project, we implement quantum transport simulations of twist angle disorder in a tight-binding model of twisted bilayer graphene. We aim to study quantitatively the effects of twist angle disorder on the wide-junction conductance of twisted bilayer graphene, and to understand the causes of these effects. 

## Objectives and questions:

* Simulate large mesoscopic devices of twisted bilayer graphene attached to two electrodes in a wide junction
* Generate parametrized ensembles with disordered samples of twisted bilayer graphene. We want these disorderd samples to be similar in shape to realistic crystals. Thus, a sample should be broken into a set of $N_d$ twist angle domains with smooth boundaries and with "bubble-like" shape. The area of the twist angle domains should be $\geq10\times10$ nm  
* Calculate the wide-junction conductance (at a range of energies containing the moiré spectrum of magic-angle twisted bilayer graphene) of each sample and average the result in each ensemble
* Calculate the spectrum of near-magic angle twisted bilayer graphene
* Study the effects of disorder and their relation to the spectrum


  ## Results:
  
* The results of this work were published in Nanotechnology: https://iopscience.iop.org/article/10.1088/1361-6528/ad90ea
* The code and data used to produced the figures in our publication can be found in the following zenodo repository: https://doi.org/10.5281/zenodo.10886941
* We find that the overall effect of twist angle disorder is to suppress and smear in energy the wide-junction conductance. However, we establish a characteristic feature of twist angle disorder that distinguishes it from other types of disorder: it is significantly stronger in the hole bands than in the electron bands
* We establish the reason for the asymmetric effect of twist angle disorder in the conductance: the density of states at angles bellow the magic angle is also asymmetric and is more smeared and less pronounced in the hole bands.
* The asymmetric effect of twist angle disorder can be expected to survive other types of disorder in realistic samples (we studied on-site electrostatic potential disorder, effects of temperature bellow 5K, Hartree interactions and lattice corrugation) 
* We propose that the asymmetric effect of twist angle disorder in near-magic angle twisted bilayer graphene can be used to characterize disorder in real devices

## Remarks

- The folder `./code` contains code and Jupyter notebooks to perform the quantum transport simulations. The functionality of each file is the following:
    - The notebook `./code/generate_disorder_domains.ipynb` generates samples with randomized twist angle domains. It also contains tools for the statistical analysis of samples and the optimization of the parameters to generate the ensembles. 
    - The python script `./code/constants.py` contains the constants used in all simulations
    - The python script `./code/create_sample.py` containts the class "TBGSample" that creates objects that represent finite devices of twisted bilayer graphene attached to two semi-infinite leads in their wide junction
    - The notebook `./code/calculate_conductance.ipynb` has the functionality of calculating the conductance matrix of a TBGSample. It can also be used to visualize devices and twist angle domains
    - The notebook `./code/calculate_spectrum.ipynb` has the functionality of calculating the spectrum and density of states of twisted bilayer graphene 
    - The python script `./code/Descartes_fix/patch.py` contains a patch necessary to fix an **incopatibility issue**. See the section "Installing the environment" bellow
    - The folder `./data/spectrum/` has the data with the spectra and density of states of several twist angles and interlayer hopping amplitudes. These are the parameters used in our paper
    - The folder `./data/selected_polygons/` has the polygons for the ensembles with $N_d=3,5$ and 7 twist angle domains. Each ensemble has 20 samples with randomized domains
    - The file `./twist-disorder-environment.yml` contains the environment with all the packages needed to run these simulations
      
## Installing the environment

To install the environment, use the `conda`  environment management system. Run the following command:

`conda env create -f twist-disorder-environment.yml`

Due to an incopatibility between packages **shapely** and **descartes**, a fix is needed for some plots used in our paper (the 3D projections of the finite system). The way I solved this problem, is by manually changing a few functions in the `descartes` package. All the fixes are included in the file `./code/Descartes_fix/patch.py`. After installing the environment with the `conda` command above, go to the file in which `descartes` is installed (typically located in `/home/user/anaconda3/envs/twist-disorder-environment/lib/python3.8/site-packages/descartes/patch.py`) and replace this file with the one from the fix (`./code/Descartes_fix/patch.py` in this repository)
