import re

import qnetvo


def test_version():

    version_str = qnetvo.__version__

    assert isinstance(version_str, str)
    assert re.fullmatch(r"0\.1\.\d+", version_str) != None
