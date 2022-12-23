from .version import __version__
from .utilities import *
from .network_nodes import *
from .network_ansatz import *
from .ansatz_library import *
from .qnodes import *
from .postprocessing import *
from .information import *
from .gradient_descent import *
from .cost import *

# adding the quantum channels to "default.mixed" device
from pennylane.devices import DefaultMixed

DefaultMixed.operations.update(["two_qubit_depolarizing", "colored_noise"])
