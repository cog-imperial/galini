import mpmath


def pytest_sessionstart(session):
    mpmath.mp.dps = 20  # 20 decimal places precision
