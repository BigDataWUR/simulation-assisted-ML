# Simulation-assisted machine learning for operational digital twins

Implementation of the paper [Simulation-assisted machine learning for operational digital twins](https://www.sciencedirect.com/science/article/pii/S1364815221003169). In this paper we propose a method to make predictions overcoming the following problems:
- Not enough data exist to train machine learning models
- No data exist to calibrate process-based models in a new location
- Data exist but they are on different resolution than what machine learning and process-based models expect


We evaluate this method with a case study of pasture nitrogen response rate prediction in New Zealand. The evaluation includes different microclimates, scenarios of sampled/unsampled locations, and different amounts of data included in training. For the prediction models we used random forest and tuned it with bayesian optimization.

# Setup 
Before running the code, change the paths defined in the ```main``` function inside **run.py** to point to the simulated/weather/nitrogen response rate data, and also where to save the results.

Given the following example paths:
- ```scenario_name```: NZ_Clover
- ```preprocessing_results_path```: preprocessing_results_path
- ```ml_results_path```: ml_results

the following directory hierarchies will be created:

----preprocessing_results_path/NZ_Clover  
    |    
    ----global    
    |   |    
    |   ----known    
    |   |   |    
    |   |   ----train    
    |   |       |    
    |   |       ----Clim1.csv    
    |   |       |    
    |   |       ----Clim2.csv    
    |   |       |    
    |   |       ---- ....    
    |   |    
    |   |----unknown    
    |           |    
    |           ----train    
    |               |    
    |               ----Clim1.csv    
    |               |    
    |               ----Clim2.csv    
    |               |    
    |               ---- ....    
    |    
    ----regional  
    |   |    
    |   ----known    
    |   |   |    
    |   |   ----train    
    |   |       |    
    |   |       ----Clim1.csv    
    |   |       |    
    |   |       ----Clim2.csv    
    |   |       |    
    |   |       ---- ....    
    |   |    
    |   |----unknown    
    |           |    
    |           ----train    
    |               |    
    |               ----Clim1.csv    
    |               |    
    |               ----Clim2.csv    
    |               |    
    |               ---- ....    
    |    
    ----local  
    |   |    
    |   ----known    
    |   |   |    
    |   |   ----train    
    |   |       |    
    |   |       ----Clim1.csv    
    |   |       |    
    |   |       ----Clim2.csv    
    |   |       |    
    |   |       ---- ....    
    |   |    
    |   |----unknown    
    |           |    
    |           ----train    
    |               |    
    |               ----Clim1.csv    
    |               |    
    |               ----Clim2.csv    
    |               |    
    |               ---- ....    
    |    
    ----location    
        |    
        ----test    
            |    
            ----Clim1.csv    
            |    
            ----Clim2.csv    
            |    
            ---- ....    
     
----ml_results    
    |    
    ----NZ_Clover.zip    
