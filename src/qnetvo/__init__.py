from qnetvo.utilities import *
from qnetvo.network_ansatz import *
from qnetvo.ansatz_library import *
from qnetvo.qnodes import *
from qnetvo.postprocessing import *
from qnetvo.information import *
from qnetvo.gradient_descent import *
from qnetvo.cost import *

# adding the quantum channels to "default.mixed" device
from pennylane.devices import DefaultMixed

DefaultMixed.operations.update(["two_qubit_depolarizing", "colored_noise"])
