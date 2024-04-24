import argparse
import logging
import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from combined.camera import Camera

import pybullet as p
from combined.config import (
    ARUCO_TEXTURE,
    ARUCO_URDF,
    IMG_HALF,
    IMG_SIDE,
    LAMBDA,
    SIMULATION_URDF,
    START_POSITION,
    TARGET_POSITION,
    eefLinkIdx,
    jointIndices,
    logTime,
    simulation_steps,
)


class FeatureExtractor:
    def __init__(self, body_id: int):
        self.body_id = body_id
        self.detector = self._create_detector()
        self.camera = Camera(imgSize=[IMG_SIDE, IMG_SIDE])

    @staticmethod
    def _create_detector() -> cv2.aruco.ArucoDetector:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector

    def update_camera_position(self):
        linkState = p.getLinkState(self.body_id, linkIndex=8)
        xyz = linkState[0]
        quaternion = linkState[1]
        rotation = p.getMatrixFromQuaternion(quaternion)
        rotation = np.reshape(np.array(rotation), (3, 3))
        self.camera.set_new_position(xyz, rotation)

    def get_features(self) -> cv2.typing.MatLike:
        self.update_camera_position()
        img = self.camera.get_frame()
        corners, _, _ = self.detector.detectMarkers(img)
        return corners


def computeInteractionMatrix(Z: float, sd0: np.ndarray) -> np.ndarray:
    L = np.zeros((8, 4))
    for idx in range(4):
        x = sd0[2 * idx, 0]
        y = sd0[2 * idx + 1, 0]
        L[2 * idx] = np.array([-1 / Z, 0, x / Z, y])
        L[2 * idx + 1] = np.array([0, -1 / Z, y / Z, -x])
    return L


def get_feature_vector(corners: cv2.typing.MatLike) -> np.ndarray:
    sd0 = np.reshape(np.array(corners[0][0]), (8, 1))
    sd0 = np.array([(s - IMG_HALF) / IMG_HALF for s in sd0])
    return sd0


def collect_metadata(
    box_id: int, position_list: List[List[float]], orientation_list: List[List[float]]
):
    linkState = p.getLinkState(box_id, linkIndex=8, computeLinkVelocity=1)
    position, quaternion = linkState[0], linkState[1]
    orientation = np.degrees(p.getEulerFromQuaternion(quaternion))
    position_list.append(position)
    orientation_list.append(orientation)


def plot_metadata(
    position_list: List[List[float]],
    orientation_list: List[List[float]],
    time: np.ndarray,
    figsize: Tuple = (10, 6),
):
    position_list = np.array(position_list)
    orientation_list = np.array(orientation_list)
    fig, axs = plt.subplots(nrows=2, figsize=figsize)
    fig.suptitle('Movement trajectory')
    axs[0].plot(time, position_list[:,0], label="X")
    axs[0].plot(time, position_list[:,1], label="Y")
    axs[0].plot(time, position_list[:,2], label="Z")
    axs[0].legend()
    axs[0].set_title("XYZ")
    axs[0].set_xlabel('time, sec.')
    axs[0].set_ylabel('position')

    axs[1].plot(time, orientation_list[:,0], label="Roll")
    axs[1].plot(time, orientation_list[:,1], label="Pitch")
    axs[1].plot(time, orientation_list[:,2], label="Yaw")
    axs[1].set_title("Rotation")
    axs[1].legend()
    axs[1].set_xlabel('time, sec.')
    axs[1].set_ylabel('angle, deg.')
    plt.show(block=True)

def init_enviroment(verbose: bool = True) -> int:
    options = (
        "--background_color_red=1 --background_color_blue=1 --background_color_green=1"
    )
    if verbose:
        options += " --debug"

    p.connect(p.GUI, options=options)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=-90,
        cameraPitch=-89.999,
        cameraTargetPosition=[0.5, 0.5, 0.6],
    )
    p.setGravity(0, 0, -10)
    box_id = p.loadURDF(SIMULATION_URDF, useFixedBase=True)
    c = p.loadURDF(ARUCO_URDF, (0.5, 0.5, 0.0), useFixedBase=True)
    x = p.loadTexture(ARUCO_TEXTURE)
    p.changeVisualShape(c, -1, textureUniqueId=x)
    return box_id


def start_simulation(verbose: bool = True):
    box_id = init_enviroment(verbose=verbose)
    feature_extractor = FeatureExtractor(body_id=box_id)
    numJoints = p.getNumJoints(box_id)
    positions: List[List[float]] = []
    orientations: List[List[float]] = []

    if verbose:
        for idx in range(numJoints):
            logging.info(
                f"{idx} {p.getJointInfo(box_id, idx)[1]} {p.getJointInfo(box_id, idx)[12]}"
            )
        (
            logging.info("Running with numpy backend.")
            if p.isNumpyEnabled()
            else logging.warning("Numpy is not not available! Perfomance may degrade.")
        )

    p.setJointMotorControlArray(
        bodyIndex=box_id,
        jointIndices=jointIndices,
        targetPositions=TARGET_POSITION,
        controlMode=p.POSITION_CONTROL,
    )
    for _ in range(simulation_steps):
        p.stepSimulation()
    corners = feature_extractor.get_features()
    if corners and len(corners) > 0 and len(corners[0]) > 0:
        s_target = get_feature_vector(corners=corners)
    else:
        logging.critical("No desired corners detected")

    # go to the starting position
    p.setJointMotorControlArray(
        bodyIndex=box_id,
        jointIndices=jointIndices,
        targetPositions=START_POSITION,
        controlMode=p.POSITION_CONTROL,
    )
    for _ in range(simulation_steps):
        p.stepSimulation()
    collect_metadata(box_id=box_id, position_list=positions, orientation_list=orientations)

    camCount = 0
    w = np.zeros((4, 1))
    for t in logTime[1:]:
        p.stepSimulation()
        collect_metadata(
            box_id=box_id, position_list=positions, orientation_list=orientations
        )
        camCount += 1
        if camCount == 5:
            camCount = 0
            corners = feature_extractor.get_features()
            if corners and len(corners) > 0 and len(corners[0]) > 0:
                s0 = get_feature_vector(corners=corners)
                Z0 = feature_extractor.camera.eye_position[2]
                L0 = computeInteractionMatrix(Z0, s0)
                L0T = np.linalg.inv(L0.T @ L0) @ L0.T
                e = s0 - s_target
                w = -LAMBDA * L0T @ e
            else:
                logging.critical(
                    f"No corners detected on {t}/{len(logTime[1:])} timestamp"
                )

        jStates = p.getJointStates(box_id, jointIndices=jointIndices)
        jPos = [state[0] for state in jStates]
        (linJac, angJac) = p.calculateJacobian(
            bodyUniqueId=box_id,
            linkIndex=eefLinkIdx,
            localPosition=[0, 0, 0],
            objPositions=jPos,
            objVelocities=[0, 0, 0, 0],
            objAccelerations=[0, 0, 0, 0],
        )

        J = np.block([[np.array(linJac)], [np.array(angJac)[2, :]]])
        dq = (np.linalg.inv(J) @ w).flatten()[[1, 0, 2, 3]]
        dq[2] = -dq[2]
        dq[3] = -dq[3]
        p.setJointMotorControlArray(
            bodyIndex=box_id,
            jointIndices=jointIndices,
            targetVelocities=dq,
            controlMode=p.VELOCITY_CONTROL,
        )

    p.disconnect()
    plot_metadata(position_list=positions, orientation_list=orientations, time=logTime)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Allow printing additional logging.",
    )
    args = parser.parse_args()
    return args


def main():
    logger = logging.getLogger("aruco")
    logger.setLevel(logging.INFO)
    args = _parse_args()
    start_simulation(verbose=args.verbose)


if __name__ == "__main__":
    main()
