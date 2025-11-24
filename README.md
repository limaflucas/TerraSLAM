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

### Using the worlds
This project has two different worlds to be used.

As the project starts running, the matplotlib window will pop-up. To **pause** the execution and interact with the plots, just **press the letter p**. To **resume** the execution, **press p again**.

#### City world
In the city world the robot will move in a circular way. To enable this motion, comment line 92 and uncomnent lines 89 and 90

#### Multi-track testing world
In this world we have four different tracks and each one of them will test a different aspect of EKF-SLAM algorithm.

To enable the robot motion in each track, do the following:
- **Track 01 and 04:** these tracks have: multiple tags and no obstacles (track 01) and just one tag at the end (track 04).
      1. Place the robot at the left-most corner
      2. Comment lines 89, 90
      3. Uncomment lines 92 and 228-237
- **Track 02:** this track has uneven terrain.
      1. Place the robot at the left-most corner
      2. Comment lines 89, 90
      3. Uncomment lines 92 and 240-249
- **Track 03:** this track has obstructions and some occlusion
      1. Place the robot at the left-most corner
      2. Comment lines 89, 90
      3. Uncomment lines 92 and 252-261
