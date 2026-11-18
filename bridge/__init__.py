"""Alpamayo–CARLA Bridge.

Connects NVIDIA's Alpamayo VLA model to the CARLA simulator for
autonomous driving research and visualization.
"""

from .alpamayo_inference import AlpamayoInference, AlpamayoOutput
from .driver import CarlaBridge, DriveStep
from .ego_motion import EgoState
from .observation import CameraFrame, ObservationBuffer
from .trajectory_controller import TrajectoryController, VehicleCommand
from .utils import load_config

__version__ = "0.2.0"
__all__ = ["AlpamayoInference", "AlpamayoOutput", "CarlaBridge", "DriveStep", "EgoState",
           "CameraFrame", "ObservationBuffer", "TrajectoryController", "VehicleCommand", "load_config"]
