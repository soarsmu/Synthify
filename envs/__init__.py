from .cartpole import cartpole
from .pendulum import pendulum
from .satellite import satellite
from .dcmotor import dcmotor
from .tape import tape
from .magnetic_pointer import magnetic_pointer
from .suspension import suspension
from .biology import biology
from .cooling import cooling
from .quadcopter import quadcopter
from .self_driving import self_driving
from .car_platoon_4 import car_platoon_4
from .car_platoon_8 import car_platoon_8
from .lane_keeping import lane_keeping
from .oscillator import oscillator

ENV_CLASSES = {
    "cartpole": cartpole,
    "pendulum": pendulum,
    "satellite": satellite,
    "dcmotor": dcmotor,
    "tape": tape,
    "magnetic_pointer": magnetic_pointer,
    "suspension": suspension,
    "biology": biology,
    "cooling": cooling,
    "quadcopter": quadcopter,
    "self_driving": self_driving,
    "car_platoon_4": car_platoon_4,
    "car_platoon_8": car_platoon_8,
    "lane_keeping": lane_keeping,
    "oscillator": oscillator,
}
