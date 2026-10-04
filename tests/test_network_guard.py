import socket

import pytest


def test_tests_cannot_reach_the_internet():
    with pytest.raises(RuntimeError, match="network connection"):
        socket.create_connection(("finnhub.io", 443), timeout=1)
