from context import QNetOptimizer as QNopt
from pennylane import numpy as np
from datetime import datetime
from qiskit import IBMQ

provider = IBMQ.load_account()

prep_nodes = [QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0)]
meas_nodes = [
    QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
    QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
]

local_chsh_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

dev_ibm_belem = {
    "name": "qiskit.ibmq",
    "shots": 4000,
    "backend": "ibmq_belem",
    "provider": provider,
}

ibm_belem_chsh_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, dev_kwargs=dev_ibm_belem)

rand_settings = local_chsh_ansatz.rand_scenario_settings()

chsh_cost = QNopt.chsh_inequality_cost(local_chsh_ansatz)

natural_grad = QNopt.chsh_natural_grad(ibm_belem_chsh_ansatz, diff_method="parameter-shift")

opt_dict = QNopt.gradient_descent(
    chsh_cost, rand_settings, step_size=0.2, num_steps=20, sample_width=1, grad_fn=natural_grad
)

datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

filename = "script/data/simple_chsh_opt_ibm_belem_natural_gradient_" + datetime_ext

QNopt.write_optimization_json(opt_dict, filename)
