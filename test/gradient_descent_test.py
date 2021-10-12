import pytest
import re
from pennylane import numpy as np
import tensorflow as tf

from context import QNetOptimizer as QNopt


class TestGradientDescent:
    def test_quadratic_cost(self):
        cost = lambda x: x ** 2
        opt_dict = QNopt.gradient_descent(cost, 2.0, num_steps=50, step_size=0.1, verbose=False)

        assert np.isclose(opt_dict["opt_score"], 0, atol=1e-6)
        assert np.isclose(opt_dict["opt_settings"], 0, atol=1e-4)
        assert opt_dict["samples"] == [0, 25, 49]
        assert len(opt_dict["scores"]) == 3
        assert len(opt_dict["settings_history"]) == 51

        assert len(opt_dict["step_times"]) == 3
        assert opt_dict["step_size"] == 0.1

        opt_datetime_match = re.match(r"\d+-\d+-\d+T\d\d:\d\d:\d\dZ", opt_dict["datetime"])
        assert bool(opt_datetime_match)

        # tensor flow interface
        tf_opt_dict = QNopt.gradient_descent(
            cost, tf.Variable(2.0), num_steps=50, step_size=0.1, verbose=False, interface="tf"
        )

        assert np.isclose(tf_opt_dict["opt_score"], 0, atol=1e-6)
        assert np.isclose(tf_opt_dict["opt_settings"], 0, atol=1e-4)
        assert tf_opt_dict["samples"] == [0, 25, 49]
        assert len(tf_opt_dict["scores"]) == 3
        assert len(tf_opt_dict["settings_history"]) == 50

        with pytest.raises(ValueError, match='Interface "jax" is not supported.'):
            QNopt.gradient_descent(
                cost, tf.Variable(2.0), num_steps=50, step_size=0.1, verbose=False, interface="jax"
            )