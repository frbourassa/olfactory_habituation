# Manifold learning for olfactory habituation to strongly fluctuating backgrounds

Numerical simulations of olfactory habituation models for the manuscript

> François X. P. Bourassa, Paul François, Gautam Reddy, Massimo Vergassola. "Manifold learning for olfactory habituation to strongly fluctuating backgrounds", [submitted], 2025. 

bioRxiv preprint: [https://doi.org/10.1101/2025.05.26.656161](https://doi.org/10.1101/2025.05.26.656161)

## Outline of the code

**Jupyter notebooks and scripts in the main folder**: simulations generating main figure results. 

- `turbulent_habituation_recognition.ipynb`: notebook with a sample run and performance characterization of the different habituation models on a background with turbulent statistics, as in figure 4 (also used for figure S3). 
- `non-gaussian_habituation_recognition.ipynb`: notebook with a sample run on the background with weakly non-Gaussian statistics of figure 3 (also used for figure S3). 
- `manifold_learning_filtering_tradeoffs.py`: plots of the analytical results on predictive filtering vs manifold learning tradeoffs (figures 1 and S1). 
- `run_performance_recognition.py`, `run_performance_dimensionality.py`: main scripts to launch the numerical experiments of figures 2 (and S2) and 5 (and S12), respectively, comparing the performance of various habituation models for new odor recognition. 
- `analyze_comparison_results.py`, `analyze_dimensionality_results.py`: scripts to analyze the results of these simulation results and generate the summary statistics shown in the figures.
- `nonlinear_osn_turbulent_illustrations.ipynb`: example habituation simulations and nonlinear manifolds resulting from saturating OSN response functions (figures 6 and S14). 
- `run_performance_nl_osn.py`, `analyze_nl_osn_results.py`: scripts to run multiple habituation and new odor recognition tests with nonlinear OSN responses (figures 6 and S14). 

**Modules in `modelfcts`**: 
- `average_sub`, `biopca`, `ideal`, `ibcm`: implementation of the different habituation models and functions to integrate them numerically. 
- `tagging`: odor tag (Kenyon cell layer) computation.
- `ibcm_analytics`, `pca_analytics`: analytical results for the IBCM and BioPCA networks. 
- `backgrounds`, `distribs`: implementation of different stochastic background processes. 
- `nonlin_adapt_osn`: background generation and update functions with the nonlinear OSN response model and OSN adaptation. 
- `checktools`: functions to test different parts of the models and code. 

**Modules in `simulfcts`**: functions used by the main scripts to run numerical simulations, using parallel processing, saving results to disk, etc. 
- `analysis`: functions used by the main `analyze...` scripts, to process simulation results. 
- `habituation_recognition`: functions to set up and launch parallel simulations of the various habituation models. 
- `habituation_recognition_lambda`: simulation functions to re-run the same simulation for various $\Lambda$ scales. 
- `habituation_recognition_nonlin_osn`: functions to set up and launch multiple simulations with nonlinear OSNs. 
- `idealized_recognition`: functions to compute ideal/optimal habituation model outputs, especially those that would have occured in existing (saved) simulations. 
- `plotting`: auxiliary plotting functions.  

**Code in the `supplementary_scripts` folder**: main simulation code for supplementary results. 
- `autocorrelation_turbulent.py`: to compute the autocorrelation function of the turbulent odor concentration process, shown in figure S1C. 
- `importance_thirdmoment_ibcm.ipynb`: example simulation of the IBCM model on Gaussian versus weakly non-Gaussian backgrounds (figure S5). 
- `lognormal_habituation_recognition.ipynb`: example simulation and performance characterization of the different habituation models on a background with log-normal concentration statistics, for figure S6. 
- `toymodel_habituation_recognition.ipynb`: detailed analysis of habituation simulations on a toy two-odor background model, described in the supplementary materials and figures S4 and S7; in particular, analysis of ther IBCM model convergence time. 
- `convergence_time_scales_non-gaussian.ipynb`: sample simulations to compare convergence times to analytical predictions in the IBCM and BioPCA models (figure S8). 
- `convergence_ibcm_turbulent.ipynb`, `run_ibcm_convergence_turbulent.py`, `run_biopca_convergence_turbulent.py`: analysis of convergence time of the IBCM and BioPCA models as a function of various parameters (figure S9). Run the Python scripts first, since the notebook plots some of the large simulation results. 
- `check_gaussian_noise_robustness.ipynb`: example simulation of the different habituation models in the presence of OSN noise (figure S10). 
- `run_performance_lambda.py`, `run_performance_noise.py`, `run_performance_w_norm_choice.py`: scripts to launch the simulations and new odor recognition tests of figures S16 (as a function of $\Lambda$), S10 (robustness against OSN noise), and S11 (W norm choice), respectively. 
- `analyze_lambda_results.py`, `analyze_noise_results.py`, `analyze_w_norm_results.py`: scripts to analyze these simulation results and generate the summary statistics shown in the figures. 
- `turbulent_habituation_test_w_norms.ipynb`: notebook to run example habituation simulations with alternate $p, q$-norms in the $W$ update rule, as in figure S11. 
- `correlated_odors_turbulent.ipynb`, `run_performance_correlation.py`, `analyze_correlation_results.py`: run habituation and analyze odor recognition performance in the presence of correlated background odors (figure S13). 
- `nonlinear_adaptation_osn_turbulent.ipynb`: example simulation with OSN adaptation (figure S15). 
- `run_adaptation_performance_tests.py`: run multiple habituation simulations and assess new odor recognition performance with OSN adaptation (figure S15D). 
- `si2019_hill_tanh_distribution_fits.ipynb`: fitting the empirical OSN-odor affinities distribution from Si et al., 2019 (used in figures 6 and S14-15)
- `orthogonality_odor_vectors.ipynb`: analysis of odor vector distributions and illustration of non-orthogonality and non-negativity of the OSN inputs. 


**Code in the `tests` folder**: scripts to test different parts of the numerical implementation of habituation models and background processes. 


**Code in the `final_plotting` folder**: code to produce the final main figures from the simulation results generated by main and secondary scripts. 


