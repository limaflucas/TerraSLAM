# TerraSLAM

## Instructions
This is a complete self-contained project. It already has all dependencies required to run, needing just a few steps to assure the proper configuration.

### Confiring the Webots emulator
1. Open your Webots emulator
2. Open Preferences > Python command
3. Replace the existing information to: path/to/project/root/TerraSLAM/controllers/venv/bin/python
4. Press OK

### Gathering all dependencies
1. Activate the python virtual environment running:

``` shell
source /path/to/project/TerraSLAM/controllers/venv/bin/activate
```

2. Install the required dependencies

``` shell
cd /path/to/project/TerraSLAM
```

``` shell
pip install -r requirements.txt
```

3. Return to Webots simulator and have fun 😀

