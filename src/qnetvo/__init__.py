from qnetvo.utilities import *
from qnetvo.network_ansatz import *
from qnetvo.ansatz_library import *
from qnetvo.information import *
from qnetvo.cost.qnodes import *
from qnetvo.cost.postprocessing import *
from qnetvo.cost.I_3322_bell_inequality import *
from qnetvo.cost.mermin_klyshko_inequality import *
from qnetvo.cost.nlocal_chain_bell_inequality import *
from qnetvo.cost.nlocal_star_bell_inequality import *
from qnetvo.cost.magic_squares_game import *
from qnetvo.cost.chsh_inequality import *
from qnetvo.cost.linear_inequalities import *
from qnetvo.cost.mutual_info import *
from qnetvo.gradient_descent import *

# adding the quantum channels to "default.mixed" device
from pennylane.devices import DefaultMixed

DefaultMixed.operations.update(["two_qubit_depolarizing", "colored_noise"])
