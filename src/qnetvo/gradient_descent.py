import pennylane as qml
from datetime import datetime
import time


def gradient_descent(
    cost,
    init_settings,
    num_steps=150,
    step_size=0.1,
    sample_width=25,
    grad_fn=None,
    verbose=True,
    interface="autograd",
):
    """Performs a numerical gradient descent optimization on the provided ``cost`` function.
    The optimization is seeded with (random) ``init_settings`` which are then varied to
    minimze the cost.

    :param cost: The cost function to be minimized with gradient descent.
    :type cost: function

    :param init_settings: A valid input for the cost function.
    :type init_settings: array-like[float]

    :param num_steps: The number of gradient descent iterations, defaults to ``150``.
    :type num_steps: int, optional

    :param step_size: The learning rate for the gradient descent, defaults to ``0.1``.
    :type step_size: float, optional

    :param sample_width: The number of steps between "sampled" costs which are printed/returned
        to user, defaults to ``25``.
    :type sample_width: int, optional

    :param grad_fn: A custom gradient function, default to ``None`` which applies the standard numerical gradient.
    :type grad_fn: function, optional

    :param verbose: If ``True``, progress is printed during the optimization, defaults to ``True``.
    :type verbose: bool, optional

    :param interface: Specifies the optimizer software either ``"autograd"`` or ``"tf"`` (TensorFlow).
    :type interface: string, default ``"autograd``"

    :return: Data regarding the gradient descent optimization.
    :rtype: dictionary, contains the following keys:

        * **opt_score** (*float*) - The maximized reward ``-(min_cost)``.
        * **opt_settings** (*array-like[float]*) - The setting for which the optimum is achieved.
        * **scores** (*array[float]*) - The list of rewards sampled during the gradient descent.
        * **samples** (*array[int]*) - A list containing the iteration for each sample.
        * **settings_history** (*array[array-like]*) - A list of all settings found for each
          intermediate step of gradient descent
        * **datetime** (*string*) - The date and time in UTC when the optimization occurred.
        * **step_times** (*list[float]*) - The time elapsed during each sampled optimization step.
        * **step_size** (*float*) - The learning rate of the optimization.

    .. warning::

        The ``gradient_descent`` function minimizes the cost function, however, the general
        use case within this project is to maximize the violation of a Bell inequality.
        The maximization is assumed within ``gradient_descent`` and is applied by multiplying
        the cost by (-1). This is an abuse of function naming and will be resolved in a future
        commit by having ``gradient_descent`` return the minimized cost rather than the maximized
        reward. The resolution is to wrap ``gradient_descent`` with a ``gradient_ascent`` function
        which maximizes a reward function equivalent to ``-(cost)``.

    :raises ValueError: If the ``interface`` is not supported.
    """

    if interface == "autograd":
        opt = qml.GradientDescentOptimizer(stepsize=step_size)
    elif interface == "tf":
        from .lazy_tensorflow_import import tensorflow as tf

        opt = tf.keras.optimizers.SGD(learning_rate=step_size)
    else:
        raise ValueError('Interface "' + interface + '" is not supported.')

    settings = init_settings
    scores = []
    samples = []
    step_times = []
    settings_history = [init_settings]

    start_datetime = datetime.utcnow()
    elapsed = 0

    if grad_fn is None:
        grad_fn = qml.grad(cost, argnum=0)

    # performing gradient descent
    for i in range(num_steps):
        if i % sample_width == 0:
            score = -(cost(settings))
            scores.append(score)
            samples.append(i)

            if verbose:
                print("iteration : ", i, ", score : ", score)

        start = time.time()
        if interface == "autograd":
            settings = opt.step(cost, settings, grad_fn=grad_fn)
        elif interface == "tf":
            # opt.minimize updates settings in place
            tf_cost = lambda: cost(settings)
            opt.minimize(tf_cost, settings)

        elapsed = time.time() - start

        if i % sample_width == 0:
            step_times.append(elapsed)

            if verbose:
                print("elapsed time : ", elapsed)

        settings_history.append(settings)

    opt_score = -(cost(settings))
    step_times.append(elapsed)

    scores.append(opt_score)
    samples.append(num_steps)

    return {
        "datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "opt_score": opt_score,
        "opt_settings": settings,
        "scores": scores,
        "samples": samples,
        "settings_history": settings_history,
        "step_times": step_times,
        "step_size": step_size,
    }
