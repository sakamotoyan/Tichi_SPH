# Tichi_SPH

## Requirements

* Python 3.7 or higher
* numpy
* taichi 0.8.0 or higher
* Vulkan https://vulkan.lunarg.com/sdk/home

(run `pip install taichi` to install)
> For more about Taichi, please refer to https://github.com/taichi-dev/taichi

## Run this project
* `python main.py` to run
### command line parameters
-c <configFile> or --configFile <configFile>: specify the path of parameters config file
-s <scenarioFile> or --scenarioFile <scenarioFile>: specify the path of simulation scenario file
### example
```
python main.py -c config_example\config_3d.json -s scenario_example\3d_bunny.json

multiphase example:
python main.py -c config_example\config_3d.json -s scenario_example\two_phases.json
```