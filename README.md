# Reproducibility

A Docker file is present to build the needed container (virtual envs are kind of "vintage" nowadays).
There are two options:
- [JUST RUN] run either ```scripts/start.sh``` (unix) or ```scripts/start.bat``` (windows) to build the container and launch ```calculate_et0.py```. If you want to run something different, you should either: (i) modify the scripts or (ii) once the container is running, run ```docker exec et0_calculation [your command]```, in this case ```et0_calculation``` is the name of the container and ```[your command]``` could be something like ```python file_name.py --param_name param_value``` or ```bash another_script.sh```. This option is usually use in deployment, not in dev.
- [RUN & DEBUG] open vscode, which should suggest you to open the project in the devcontainer. Here, you can both run and debug each file through the vscode interface. In the container, there are some plugins installed that both help to maintain a "good-quality" code (e.g., black formatter) and are useful to develop/see results (csv reader). (To notice: if you open a terminal, you will find yourself inside the container, it is like your world is just the container.)

# Launch commands

```
git clone https://github.com/josephgiovanelli/mo-importance.git
cd mo-importance/
sudo chmod 777 scripts/*
sudo ./scripts/start.sh 0.0.1
```
# Machine assignments

```
.51: stratified + bounds on performance
.52: no stratified + no bounds on performance
.51: stratified + no bounds on performance
```