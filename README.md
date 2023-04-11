# Reproducibility

A Docker file is present to build the needed container (virtual envs are kind of "vintage" nowadays).
There are two options:
- [JUST RUN] run either ```scripts/start.sh``` (unix) or ```scripts/start.bat``` (windows) to build the container and launch ```calculate_et0.py```. If you want to run something different, you should either: (i) modify the scripts or (ii) once the container is running, run ```docker exec et0_calculation [your command]```, in this case ```et0_calculation``` is the name of the container and ```[your command]``` could be something like ```python file_name.py --param_name param_value``` or ```bash another_script.sh```. This option is usually use in deployment, not in dev.
- [RUN & DEBUG] open vscode, which should suggest you to open the project in the devcontainer. Here, you can both run and debug each file through the vscode interface. In the container, there are some plugins installed that both help to maintain a "good-quality" code (e.g., black formatter) and are useful to develop/see results (csv reader). (To notice: if you open a terminal, you will find yourself inside the container, it is like your world is just the container.)

# Project structure

- ```.devcontainer```: configuration to instantiate a dev container w/ vscode
- ```data```: raw agro data for the analysis
- ```doc```: documentation explaining data and related semantic
- ```scripts```: starting point for reproducibility
- ```src```: source code

# data

- ```et0```: evapotranspiration every 24 hours calculated with the two formulas
- ```interp_obs```: fine grained soil moisture profiles calulated with the bilinear interpolation from the ```raw_obs```
- ```meteo```: weather data
- ```raw_obs```: samples of the water potential sensors and the related water content values (exploiting an ad-hoc retention curve)
- ```sim```: fine grained soil moisture profiles extracted from a soil & crop simulator
- ```water```: precipitation and irrigation data

# doc

- ```doc.pdf```: documentation explaining the semantic of the data (our field is ```T1 basso```)
- ```water_content_to_water_quantity```: link to a Google Sheet explaining the calulation of the water quantity from the water content (and the volume and teh density of each cell)

# scripts

- ```start.sh```: it runs the et0 calculation on Unix-like OS
- ```start.bat```: it runs the et0 calculation on Windows-like OS

# src

- ```et0```: it contains all the formulas for calculating the et0
- ```calculate_et0```: calculate et0 (details in the following section)
- ```interpolate_gp```: interpolate raw data (raw_obs) with a bilinear profile function and produce fine-grained soil moisture profiles (interp_data). You can run, for example: ```python src/interpolate.py --input-file data/raw_obs/water_content.csv --output-file data/obs/water_content.csv```
- ```utils.py```: some input and output utils


# ET0 Formulas

## Hargreaves

Still doesn't implemented:

<img width="543" alt="image" src="https://user-images.githubusercontent.com/41596745/207830808-f53a37d2-efca-423c-a0c2-dd3d0600a806.png">

## Penman Monteith

The formula is explained [here](https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation) and is encoded in the et0 package.
To calculate the et0 from meteo data, run the following python file:
```
python src/calculate_et0.py
```
