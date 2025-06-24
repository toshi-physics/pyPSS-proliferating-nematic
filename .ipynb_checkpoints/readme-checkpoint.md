These are simulation files associated with the study by Parmar et al "A Proliferating Nematic That Collectively Senses an Anisotropic Substrate".

For details about simulation parameters see the associated preprint.

For a brief intro to pseudo spectral solvers see : https://en.wikipedia.org/wiki/Pseudo-spectral_method.

How to run the model: (make sure you have python3 installed.)

Use bash file to supply parameters to the model file and run simulation.

In a terminal type:

bash run_model_friction.s model_anisotropic_friction 0.4 -0.1 1.2 1.0 50 1

To run model with asymmetric frition with epsilon=0.4, alpha=-0.1, chi=1.2, lambda=1.0, rhoseed=50, and run=1. Similarly type:

bash run_model_field.s model_external_field 0.002 -0.1 1.2 1.0 50 1

to run model with external field with Pi=0.002, alpha=-0.1, chi=1.2, lambda=1.0, rhoseed=50, and run=1.

The simulations are stored in the data directory. Use the visualisation files to view the results. The data folder currently contains first n=5 steps for both simulations.