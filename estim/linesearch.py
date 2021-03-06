
import autograd.numpy as np
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


def backtrack(f, x, g, p, a0, t, n, l, h):
    a = a0
    fk = f(x)
    for i in range(n):
        fa = f(x + a*p)
        if np.all(l <= x+a) and np.all(x+a <= h) and fa < fk + a * 0.001 * np.dot(p, g):
            return a*p
        else:
            a *= t
    else:
        return None



def linesearch(f, x0, a0, l, h, n):
    x = x0
    grad_f = grad(f)
    cd = ChangeDetect(f(x))
    for _ in range(n):
        g = grad_f(x)
        p = -g/np.linalg.norm(g)
        a = backtrack(f, x, g, p, a0, 0.5, 10, l, h)

        if a is not None:
            x += a
            if not cd.changed(f(x), 0.0001):
                return x
        else:
            return x
    else:
        return x


def backtrack1d(f, x, a0, t, n, l, h):
    a = a0
    fk = f(x)
    for i in range(n):
        fa = f(x + a)
        if l <= x+a <= h and fa < fk:
            return fa, x+a
        else:
            a *= t
    else:
        return None, None


def linesearch1d(f, x0, a0, l, h, n):
    x = x0
    cd = ChangeDetect(f(x))
    for _ in range(n):
        fa, xa = backtrack1d(f, x, a0, 0.5, 10, l, h)
        fb, xb = backtrack1d(f, x, a0, 0.5, 10, l, h)

        if fa is not None and fb is not None:
            if abs(fa) < abs(fb):
                x = xa
            else:
                x = xb
        elif fa is not None:
            x = xa
        elif fb is not None:
            x = xb
        else:
            return x

        if not cd.changed(f(x), 0.0001):
            return x
    else:
        return x