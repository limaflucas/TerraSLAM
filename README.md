# TerraSLAM
<img width="1305" height="1187" alt="image" src="https://github.com/user-attachments/assets/2e51134d-3ca4-46b3-acb8-d3d9174041f1" />


## Instructions
This is a complete self-contained project. It already has all dependencies required to run, needing just a few steps to assure the proper configuration.

### Configuring the Webots simulator
1. Open your Webots simulator
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

