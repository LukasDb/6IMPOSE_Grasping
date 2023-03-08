from models.targetmodel import TargetModel
from models.cameras.base_camera import Camera
from models.base_gripper import BaseGripper
from abc import ABC, abstractmethod, abstractproperty


class Robot(ABC):
    """ Interface for describing robot """
    endeffector: TargetModel = None
    eye_in_hand: Camera = None
    gripper: BaseGripper = None

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def set_simulate(self, state) -> None:
        pass

    @abstractmethod
    def set_lock(self, state) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @property
    @abstractmethod
    def online(self) -> bool:
        pass
