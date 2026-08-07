# Rotating_fluid

Simulates 1D axis-symmetric rotating fluid with WENO5 scheme.

To run the code:  
`./starter.sh name ngrid CFL`  
where 
  - name = {steady, Gauss, shock} are three different initial condition:  
    * steady: Initial density given by analytic equilibrium solution  
    * Gauss:  Initial angular velocity as Gauss function  
    * shock:  Initial radial velocity is discontinous  
  - ngrid : Number of physical grids (ghost grid not included)  
  - CFL : The CFL number of simulation. Stable for ~0.5  

`evolution_1D_WENO5.py` is the main code. Of course you need first install python first. It constructs initial condition and solve for evolution equations. It will create a folder and output intermediate solutions as pngs. Then `movie.sh` makes the movie.  

This code is well-tested to have 2nd order spatial convergence for smooth solution and ~0.8 order convergence with shock.  
  
Explicite formulation and numerical implementation can be found in presentation.pptx
