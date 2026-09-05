"""Small app."""
import os


def alpha(x):
    """Compute alpha from x.

    Doubles the input.
    """
    return beta(x) * 2


def beta(y):
    return y + 1


class Gamma:
    """Holder of things."""

    def run(self):
        return alpha(3)


def trailing():
    return os.getcwd()
