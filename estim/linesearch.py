import numpy as np
from autograd import grad

class ChangeDetect:
    def __init__(self, init = None):
        if init is not None:
            self.last = np.array(init)
        else:
            self.last = None

    def changed(self, newval, threshold):
        if self.last is None:
            self.last = np.array(newval)
            return True
        else:
            change = np.abs((newval - self.last)/(self.last + np.finfo(float).eps))
            self.last = np.array(newval)
            return np.any(change > threshold)


def backtrack(f, x, a0, t, n, l, h):
    a = a0
    fk = f(x)
    for i in range(n):
        fa = f(x + a)
        if (l <= x+a <= h) and fa < fk:
            return a
        else:
            a *= t
    else:
        return None


def linesearch(f, x0, a0, l, h, n):
    x = x0
    g = grad(f)
    cd = ChangeDetect(f(x))
    for _ in range(n):
        s = g(x)

        a = backtrack(f, x, -s * a0, 0.5, 10, l, h)

        if a is not None:
            x += a
            if not cd.changed(f(x), 0.001):
                return x
        else:
            return x
    else:
        return x
