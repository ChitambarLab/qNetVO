import pytest
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestPostProcessing:
    def test_parity_vector(self):
        assert np.all(QNopt.parity_vector(1) == [1, -1])
        assert np.all(QNopt.parity_vector(2) == [1, -1, -1, 1])
        assert np.all(QNopt.parity_vector(3) == [1, -1, -1, 1, -1, 1, 1, -1])

        with pytest.raises(
            ValueError,
            match="Input `n_qubits` must satisfy `n_qubits >= 1`.",
        ):
            QNopt.parity_vector(0)

    def test_even_parity_ids(self):
        assert len(QNopt.even_parity_ids(1)) == 1
        assert np.all(QNopt.even_parity_ids(1) == [0])
        assert np.all(QNopt.even_parity_ids(2) == [0, 3])
        assert np.all(QNopt.even_parity_ids(3) == [0, 3, 5, 6])

        with pytest.raises(
            ValueError,
            match="Input `n_qubits` must satisfy `n_qubits >= 1`.",
        ):
            QNopt.parity_vector(0)
