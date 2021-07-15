# NZ

The code used to produce the results of the paper. Before running it, change the paths defined in the ```main``` function inside **run.py** to point to the simulated/weather/nitrogen response rate data, and also where to save the results.

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
