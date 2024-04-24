import numpy as np


# image parameters
IMG_SIDE = 300
IMG_HALF = IMG_SIDE/2

LAMBDA = 1 / 2
# simulation parameters
simulation_steps = 100
dt = 1/240
maxTime = 8
logTime = np.arange(0.0, maxTime, dt)
SIMULATION_URDF = "combined/simple.urdf.xml"
ARUCO_URDF = "combined/aruco.urdf"
ARUCO_TEXTURE = "combined/aruco_cube.png"
START_POSITION = [0.0, 1.5708, 0.0, 0.5]
TARGET_POSITION = [0.0, 1.5708, 0.0, 0.0]


# robot parameters
jointIndices = [1,3,5,7]
eefLinkIdx = 8
