# Manifold learning for olfactory habituation to strongly fluctuating backgrounds

Numerical simulations of olfactory habituation models for the manuscript

> François X. P. Bourassa, Paul François, Gautam Reddy, Massimo Vergassola. "Manifold learning for olfactory habituation to strongly fluctuating backgrounds", [accepted, *PRX Life*], 2025. 

bioRxiv preprint: [https://doi.org/10.1101/2025.05.26.656161](https://doi.org/10.1101/2025.05.26.656161)


# Code generating the results of each figure

Final plots are produced by code in `final_plotting/`, using the results produced by the following code. 


## Main figures

### Figure 1 (Manifold learning vs predictive filtering)
- `manifold_learning_filtering_tradeoffs.py` plots the theoretical calculation results in a simple case. 

### Figure 2 (Numerical experiments of habituation with manifold learning models)
- `run_performance_recognition.py` and `analyze_comparison_results.py` launch and analyze, respectively, the numerical experiments of habituation and new odor recognition. 

### Figure 3 (Analysis of IBCM model habituation and specificity)
- `non-gaussian_habituation_recognition.ipynb` generates and analyzes a simulation example of habituation and new odor recognition in a background with weakly non-Gaussian statistics. 

### Figure 4 (IBCM and BioPCA Habituation to turbulent backgrounds)
- `turbulent_habituation_recognition.ipynb` generates and analyzes a simulation example of habituation and new odor recognition in a background with turbulent statistics. 

### Figure 5 (Recognition performance vs dimensionality and concentration)
- `run_performance_dimensionality.py` and `analyze_dimensionality_results.py` launch and analyze simulations of habituation and new odor recognition for different odor space dimensions and new odor concentrations. 

### Figure 6 (Habituation and recognition with nonlinear OSN responses)
- `nonlinear_osn_turbulent_illustrations.ipynb` produces example simulations of habituation to nonlinear olfactory backgrounds
- `run_performance_nl_osn.py` and `analyze_nl_osn_results.py` run and analyze multiple habituation and new odor recognition tests in the presence of nonlinear OSN responses. 
- `supplementary_scripts/si2019_hill_tanh_distribution_fits.ipynb` fits the empirical OR-odor affinity distribution from Si et al., 2019 used to generated odors for the nonlinear OSN model. 




## Supplementary figures

### Figure S1 (Comparing manifold learning and predictive filtering)
- `manifold_learning_filtering_tradeoffs.py` produces the detailed plots of the loss vs variance and dimensionality (S1A, S1B). 
- `supplementary_scripts/autocorrelation_turbulent.py` computes the autocorrelation function of the turbulent odor concentration process (figure S1C). 

### Figure S2 (New odor vs background recognition after habituation)
- `run_performance_recognition.py` and `run_performance_dimensionality.py`, used for Figure 2, also generate the supplementary results in Figure S2. 

### Figure S3 (Fixed point eigenvalues of the IBCM model) 
- `non-gaussian_habituation_recognition.ipynb` computes IBCM eigenvalues at the fixed point in a weakly non-Gaussian background, for S3A. 
- `supplementary_scripts/lognormal_habituation_recognition.ipynb` computes eigenvalues in a log-normal background, for S3B. 
- `turbulent_habituation_recognition.ipynb` computes eigenvalues in a turbulent background, for S3C. 

### Figure S4 (IBCM learning dynamics on a simplified background)
- `supplementary_scripts/toymodel_habituation_recognition.ipynb` provides a detailed analysis of habituation on a toy two-odor background model, described in the appendix section 6, and analyzes IBCM convergence time in particular. 

### Figure S5 (IBCM depends on higher-oder moments of the background)
- `supplementary_scripts/importance_thirdmoment_ibcm.ipynb` generates example simulations of the IBCM model on Gaussian versus weakly non-Gaussian backgrounds.  

### Figure S6 (Habituation to log-normal background statistics)
- `supplementary_scripts/lognormal_habituation_recognition.ipynb` runs example habituation and recognition simulations in log-normal background statistics. 

### Figure S7 (BioPCA learning dynamics in a simplified background)
- `supplementary_scripts/toymodel_habituation_recognition.ipynb` also includes simluations for the BioPCA model in the toy 2D background. 

### Figure S8 (Convergence time analysis of IBCM and BioPCA neurons)
- `supplementary_scripts/convergence_time_scales_non-gaussian.ipynb` runs sample simulations to compare convergence times to analytical predictions in the IBCM and BioPCA models. 

### Figure S9 (Convergence of IBCM and BioPCA vs model and turbulence parameters)
- `supplementary_scripts/run_ibcm_convergence_turbulent.py` runs multiple IBCM habituation simulations to measure convergence as a function of various model and background parameters. 
- `supplementary_scripts/run_biopca_convergence_turbulent.py` does the same, for BioPCA. 
- `supplementary_scripts/convergence_ibcm_turbulent.ipynb` runs example simulations and plot the full results of the above scripts for the analysis of convergence time in the IBCM and BioPCA models; one needs to run `run_biopca_convergence_turbulent.py` and `run_ibcm_convergence_turbulent.py` first. 

### Figure S10 (IBCM and BioPCA robustness against OSN noise )
- `supplementary_scripts/check_gaussian_noise_robustness.ipynb` runs example simulation of the different habituation models in the presence of OSN noise
- `supplementary_scripts/run_performance_noise.py` and `supplementary_scripts/analyze_noise_results.py` launch and analyze multiple simulations to test habituation and new odor recognition in the presence of OSN noise. 

### Figure S11 (Performance for various $L^p$ norms in $W$'s learning rule)
- `supplementary_scripts/run_performance_w_norm_choice.py` and `supplementary_scripts/analyze_w_norm_results.py` run and analyze multiple habituation and new odor recognition tests for various choices of $W$ learning rates based on using diffrent $L^p$ norms in the cost function. 
- `supplementary_scripts/turbulent_habituation_test_w_norms.ipynb` runs sample simulations with alternate $L^p, L^q$ norms in the $W$ update rule. 

### Figure S12 (Recognition performance vs dimensionality and concentration, supplement)
- `run_performance_dimensionality.py` and `analyze_dimensionality_results.py`, used for Fig. 5, generate the full results shown in S12.

### Figure S13 (Manifold learning in the presence of correlated odor concentrations)
- `supplementary_scripts/correlated_odors_turbulent.ipynb` runs sample habituation and odor recognition simulations for different strengths of correlations between a pair of background odor concentrations. 
- `supplementary_scripts/run_performance_correlation.py` and `supplementary_scripts/analyze_correlation_results.py` assess habituation and odor recognition performance across multiple simulations for various strengths of correlations between a pair of odors. 

### Figure S14 (Supplementary results with nonlinear OSN responses.)
The code for Fig. 6 also generates the supplementary results for strong OSN nonlinearity shown in Fig. S14: 
  - `nonlinear_osn_turbulent_illustrations.ipynb` for sample simulations at different $\epsilon$ values. 
  - `run_performance_nl_osn.py` and `analyze_nl_osn_results.py` to assess performance as a function of $\epsilon$, across multiple simulation seeds for each $\epsilon$, for different new odor concentrations. 
  - `supplementary_scripts/si2019_hill_tanh_distribution_fits.ipynb` fits the empirical odor affinity distribution. 

### Figure S15 (Impact of nonlinear OSN adaptation on manifold learning)
- `supplementary_scripts/nonlinear_adaptation_osn_turbulent.ipynb` generates an example simulation with OSN adaptation.
- `supplementary_scripts/run_adaptation_performance_tests.py` is an all-in-one script that runs multiple habituation simulations and assesses new odor recognition performance with OSN adaptation (results in Fig. S15). 
- `supplementary_scripts/si2019_hill_tanh_distribution_fits.ipynb`fits the empirical odor affinity distribution also used for nonlinear (but not adapting) OSNs in Figs. 6 and S14

### Figure S16 (Effect of the $M$ weights scaling parameter $\Lambda$)
- `supplementary_scripts/run_performance_lambda.py` and `supplementary_scripts/analyze_lambda_results.py` run and analyze a range of habituation simulations on the same background for different values of the $M$ weights scaling parameter, $\Lambda$. 




# Modules containing the core models and simulations procedures

### Modules in `modelfcts` 
Implementation of the habituation models and of the background processes. 

- `average_sub`, `biopca`, `ideal`, `ibcm`: implementation of the different habituation models and functions to integrate them numerically. 
- `tagging`: odor tag (Kenyon cell layer) computation.
- `ibcm_analytics`, `pca_analytics`: analytical results for the IBCM and BioPCA networks. 
- `backgrounds`, `distribs`: implementation of different stochastic background processes. 
- `nonlin_adapt_osn`: background generation and update functions with the nonlinear OSN response model and OSN adaptation. 
- `checktools`: functions to test different parts of the models and code. 

### Modules in `simulfcts` 
Functions used by the main scripts to run numerical simulations, using parallel processing, saving results to disk, etc.

- `analysis`: functions used by the main `analyze...` scripts, to process simulation results. 
- `habituation_recognition`: functions to set up and launch parallel simulations of the various habituation models. 
- `habituation_recognition_lambda`: simulation functions to re-run the same simulation for various $\Lambda$ scales. 
- `habituation_recognition_nonlin_osn`: functions to set up and launch multiple simulations with nonlinear OSNs. 
- `idealized_recognition`: functions to compute ideal/optimal habituation model outputs, especially those that would have occured in existing (saved) simulations. 
- `plotting`: auxiliary plotting functions.  



# Additional code

### Code in the `tests` folder
Scripts to test different parts of the numerical implementation of habituation models and background processes. 


### Code in the `final_plotting` folder
Code to produce the final main and supplementary figures from the simulation results generated by main and supplementary scripts. 


