import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from ..qnodes import global_parity_expval_qnode
from scipy.linalg import pinvh


def star_I22_fn(network_ansatz, parallel=False, nthreads=4, **qnode_kwargs):
    """Constructs a network-specific ``I22(network_settings)`` function that
    evaluates the :math:`I_{22,n}` quantity for the :math:`n`-local star network.

    The :math:`I_{22,n}` quantity is formally expressed as

    .. math::

        I_{22,n} = \\frac{1}{2^n}\\sum_{x_1,\\dots,x_n}\\langle A_{x_1}\\dots A_{x_n}B_0\\rangle,

    where :math:`x_i\\in\\{0,1\\}` and :math:`A_{x_i}` and :math:`B_{x_{n+1}}`
    are dichotomic observables.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True`` qnodes will be evaluated in separate threads. This is
                     valuable for execution on remote simulator and hardware devices.
                     Default value: ``False``.
    :type parallel: *optional* Bool

    :param nthreads: Specifies the number of threads used when ``parallel=True``.
    :type nthreads: Int

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type qnode_kwargs: *optional* dictionary

    :returns: A function callable as ``I22(*network_settings)`` that evaluates the :math:`I_{22,n}` quantity.
    :rtype: function
    """
    n = len(network_ansatz.layers[0])

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]
    network_input_x_vals = [
        static_prep_inputs + [[int(bit) for bit in np.binary_repr(x, width=n) + "0"]]
        for x in range(2**n)
    ]

    if parallel:
        from ..lazy_dask_import import dask

        star_qnodes = [
            global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(nthreads)
        ]
    else:
        star_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    def I22(*network_settings):
        I22_x_settings = [
            network_ansatz.qnode_settings(network_settings, network_inputs)
            for network_inputs in network_input_x_vals
        ]

        if parallel:
            I22_results = []
            num_batches = int((2**n) / nthreads)
            for i in range(num_batches + 1):
                start_id = i * nthreads
                end_id = start_id + nthreads if i < num_batches else 2**n

                I22_delayed_results = [
                    dask.delayed(star_qnodes[i])(settings)
                    for i, settings in enumerate(I22_x_settings[start_id:end_id])
                ]

                I22_results = math.concatenate(
                    [I22_results, dask.compute(*I22_delayed_results, scheduler="threads")]
                )
        else:
            I22_results = math.stack([star_qnode(settings) for settings in I22_x_settings])

        return math.sum(I22_results) / (2**n)

    return I22


def star_J22_fn(network_ansatz, parallel=False, nthreads=4, **qnode_kwargs):
    """Constructs a network-specific ``J22(network_settings)`` function that
    evaluates the :math:`J_{22,n}` quantity for the :math:`n`-local star network.

    The :math:`J_{22,n}` quantity is formally expressed as

    .. math::

        J_{22,n} = \\frac{1}{2^n}\\sum_{x_1,\\dots,x_n}(-1)^{\\sum_i x_i}\\langle A_{x_1}\\dots A _{x_n}B_1\\rangle,

    where :math:`x_i\\in\\{0,1\\}` and :math:`A_{x_i}` and :math:`B_{x_{n+1}}` are
    dichotomic observables.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True`` qnodes will be evaluated in separate threads. This is
                     valuable for execution on remote simulator and hardware devices.
                     Default value: ``False``.
    :type parallel: *optional* Bool

    :param nthreads: Specifies the number of threads used when ``parallel=True``.
    :type nthreads: Int

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type qnode_kwargs: *optional* dictionary

    :returns: A function callable as ``J22(*network_settings)`` that evaluates the :math:`J_{22,n}`
              quantity for the given ``network_settings``.
    :rtype: function
    """

    n = len(network_ansatz.layers[0])

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]
    network_input_x_vals = [
        static_prep_inputs + [[int(bit) for bit in np.binary_repr(x, width=n) + "1"]]
        for x in range(2**n)
    ]

    if parallel:
        from ..lazy_dask_import import dask

        star_qnodes = [
            global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(nthreads)
        ]
    else:
        star_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    def J22(*network_settings):
        J22_x_settings = [
            network_ansatz.qnode_settings(network_settings, network_inputs)
            for network_inputs in network_input_x_vals
        ]

        if parallel:
            J22_expvals = []
            num_batches = int((2**n) / nthreads)
            for i in range(num_batches + 1):
                start_id = i * nthreads
                end_id = start_id + nthreads if i < num_batches else 2**n

                J22_delayed_results = [
                    dask.delayed(star_qnodes[i])(settings)
                    for i, settings in enumerate(J22_x_settings[start_id:end_id])
                ]

                J22_expvals = math.concatenate(
                    [J22_expvals, dask.compute(*J22_delayed_results, scheduler="threads")]
                )
        else:
            J22_expvals = math.stack([star_qnode(settings) for settings in J22_x_settings])

        J22_scalars = math.stack(
            [(-1) ** (math.sum(input_vals[1][0:n])) for input_vals in network_input_x_vals]
        )

        return math.sum(J22_scalars * J22_expvals) / (2**n)

    return J22


def nlocal_star_22_cost_fn(network_ansatz, parallel=False, nthreads=4, **qnode_kwargs):
    """A network-specific constructor for the :math:`n`-local star Bell
    inequality for scenarios when all measurement devices in the star network
    have 2 inputs and 2 outputs.

    The :math:`n`-local star network Bell inequality is expressed as

    .. math::

        |I_{22,n}|^{1/n} + |J_{22,n}|^{1/n} \\leq 1

    where the quantities :math:`I_{22,n}` and :math:`J_{22,n}` are evaluated using
    functions constructed by the :meth:`qnetvo.star_I22_fn` and
    :meth:`qnetvo.star_J22_fn` methods respectively.
    The classical bound is found to be 1, but quantum systems can score as high as
    :math:`\\sqrt{2}`.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True`` qnodes will be evaluated in separate threads. This is
                     valuable for execution on remote simulator and hardware devices.
                     Default value: ``False``.
    :type parallel: *optional* Bool

    :param nthreads: Specifies the number of threads used when ``parallel=True``.
    :type nthreads: Int

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type qnode_kwargs: *optional* dictionary

    :returns: A function callable as ``nlocal_star_22_cost(*network_settings)`` that evaluates
              the cost as :math:`-|I_{22,n}|^{1/n} - |J_{22,n}|^{1/n}`.
    :rtype: function
    """

    n = len(network_ansatz.layers[0])

    I22 = star_I22_fn(network_ansatz, parallel=parallel, nthreads=nthreads, **qnode_kwargs)
    J22 = star_J22_fn(network_ansatz, parallel=parallel, nthreads=nthreads, **qnode_kwargs)

    def cost(*network_settings):
        I22_score = I22(*network_settings)
        J22_score = J22(*network_settings)

        return -(np.power(math.abs(I22_score), 1 / n) + np.power(math.abs(J22_score), 1 / n))

    return cost


def parallel_nlocal_star_grad_fn(network_ansatz, nthreads=4, natural_grad=False, **qnode_kwargs):
    """Constructs a parallelizeable gradient function ``grad_fn`` for the :math:`n`-local
    star cost function.

    The gradient of the :meth:`nlocal_star_22_cost_fn` is expressed as,

    .. math::

        -\\nabla_{\\vec{\\theta}}|I_{22,n}|^{1/n}
        - \\nabla_{\\vec{\\theta}}|J_{22,n}|^{1/n},

    where the gradient differentiates with respect to the network settings :math:`\\vec{\\theta}`.

    The parallelization is achieved through multithreading and intended to improve the
    efficiency of remote qnode execution.

    :param network_ansatz: The ansatz describing the :math:`n`-local chain network.
    :type network_ansatz: NetworkAnsatz

    :param nthreads: Specifies the number of threads used when ``parallel=True``.
    :type nthreads: Int

    :param natural_grad: If ``True``, the natural gradient is evaluated by scaling the
                         gradient by the inverse of the metric tensor.
    :type natural_grad: *optional* Bool

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``nlocal_star_grad(network_settings)``.
    :rtype: Function

    .. warning::
        Parallel gradient computation is flaky on PennyLane v0.28+. Intermittent  failures may occur.

    """

    from ..lazy_dask_import import dask

    n = len(network_ansatz.layers[0])

    star_qnodes = [
        global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(nthreads)
    ]

    I22_x_vals = [[int(bit) for bit in np.binary_repr(x, width=n) + "0"] for x in range(2**n)]
    J22_x_vals = [[int(bit) for bit in np.binary_repr(x, width=n) + "1"] for x in range(2**n)]

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]

    I22 = star_I22_fn(network_ansatz, parallel=True, nthreads=nthreads, **qnode_kwargs)
    J22 = star_J22_fn(network_ansatz, parallel=True, nthreads=nthreads, **qnode_kwargs)

    # gradient helper functions
    def _g(qnode, settings):
        return qml.grad(qnode)(settings)

    def _ng(qnode, settings):
        metric_inv = pinvh(qml.metric_tensor(qnode, approx="block-diag")(settings))
        return metric_inv @ _g(qnode, settings)

    _grad = _ng if natural_grad else _g

    def nlocal_star_grad(*network_settings):
        I22_score = I22(*network_settings)
        J22_score = J22(*network_settings)

        I22_x_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in I22_x_vals
        ]
        J22_x_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in J22_x_vals
        ]

        grad_I22_results = []
        grad_J22_results = []

        num_batches = int((2**n) / nthreads)
        for i in range(num_batches + 1):
            start_id = i * nthreads
            end_id = start_id + nthreads if i < num_batches else 2**n

            grad_I22_delayed_results = [
                dask.delayed(_grad)(star_qnodes[i], settings)
                for i, settings in enumerate(I22_x_settings[start_id:end_id])
            ]
            grad_I22_results.extend(dask.compute(*grad_I22_delayed_results, scheduler="threads"))

            grad_J22_delayed_results = [
                dask.delayed(_grad)(star_qnodes[i], settings)
                for i, settings in enumerate(J22_x_settings[start_id:end_id])
            ]
            grad_J22_results.extend(dask.compute(*grad_J22_delayed_results, scheduler="threads"))

        settings_grad = network_ansatz.zero_network_settings()

        I22_scalar = (
            -(1 / n)
            * np.power(math.abs(I22_score), (1 - n) / n)
            * math.sign(I22_score)
            * (1 / (2**n))
        )
        J22_scalar = (
            -(1 / n)
            * np.power(math.abs(J22_score), (1 - n) / n)
            * math.sign(J22_score)
            * (1 / (2**n))
        )

        for i in range(2**n):
            I22_inputs = I22_x_vals[i]
            J22_inputs = J22_x_vals[i]

            J22_sign = (-1) ** math.sum(J22_inputs[0:n])

            settings_grad += I22_scalar * network_ansatz.expand_qnode_settings(
                grad_I22_results[i], [[0] * n, I22_inputs]
            )
            settings_grad += (
                J22_sign
                * J22_scalar
                * network_ansatz.expand_qnode_settings(grad_J22_results[i], [[0] * n, J22_inputs])
            )

        return settings_grad

    return nlocal_star_grad
