"""
Lazily imports tensorflow when needed.
Throws an error if the user does not have tensorflow installed.
"""
try:
    import tensorflow

    from tensorflow import keras

except ImportError as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for this functionality and is currently installed."
        "\nTensorFlow can be installed using pip:"
        "\n\npip install tensorflow"
    ) from e
