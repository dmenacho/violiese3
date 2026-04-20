from enum import Enum, auto
from pymatlie.se3 import SE3
from utils.systems import se3_rigid_body

class Q_Space(Enum):
    """Enum for configuration space types."""

    STATE_SPACE = auto()
    LIE = auto()

class NonconformityScore(Enum):
    """Enum for nonconformity score types."""
    MAHALANOBIS = auto()
    L2 = auto()

class RobotType(Enum):
    """Enum for robot types."""
    DRONE_SE3 = auto()

ROBOT_SYSTEMS = {
    (RobotType.DRONE_SE3, Q_Space.LIE): {
    "cls": se3_rigid_body.SE3RigidBody,
    "factory": lambda inertia_matrix: se3_rigid_body.SE3RigidBody(
        inertia_matrix=inertia_matrix
    ),
    }
}

ROBOT_QSPACE_TO_FACTORY = {
    key: val["factory"] for key, val in ROBOT_SYSTEMS.items()
}


class ConfigMapper:
    _registry = {}

    @staticmethod
    def register(from_key, to_key, fn):
        ConfigMapper._registry[(from_key, to_key)] = staticmethod(fn)

    @staticmethod
    def map(from_key, to_key, q):
        key = (from_key, to_key)
        return ConfigMapper._registry[key](q)


class VelocityMapper:
    _registry = {}

    @staticmethod
    def register(from_key, to_key, fn):
        VelocityMapper._registry[(from_key, to_key)] = staticmethod(fn)

    @staticmethod
    def map(from_key, to_key, q, dq_or_v):
        key = (from_key, to_key)
        return VelocityMapper._registry[key](q, dq_or_v)


for robot_type in RobotType:
    for space in Q_Space:
        key = (robot_type, space)
        ConfigMapper.register(key, key, lambda q: q)
        VelocityMapper.register(key, key, lambda q, dq: dq)



ConfigMapper.register(
    (RobotType.DRONE_SE3, Q_Space.STATE_SPACE),
    (RobotType.DRONE_SE3, Q_Space.LIE),
    lambda q: SE3.map_q_to_configuration(q),
)

ConfigMapper.register(
    (RobotType.DRONE_SE3, Q_Space.LIE),
    (RobotType.DRONE_SE3, Q_Space.STATE_SPACE),
    lambda g: SE3.map_configuration_to_q(g),
)

VelocityMapper.register(
    (RobotType.DRONE_SE3, Q_Space.STATE_SPACE),
    (RobotType.DRONE_SE3, Q_Space.LIE),
    lambda q, dq: SE3.map_dq_to_velocity(q, dq),
)

VelocityMapper.register(
    (RobotType.DRONE_SE3, Q_Space.LIE),
    (RobotType.DRONE_SE3, Q_Space.STATE_SPACE),
    lambda q, v: SE3.map_velocity_to_dq(q, v),
)