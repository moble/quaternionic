import abc
import numpy as np
from . import jit


class QuaternionPropertiesMixin(abc.ABC):
    """Basic properties for quaternionic array class

    This abstract base class comprises the basic interface for quaternionic
    arrays, including the components (w, x, y, z), the parts (scalar, vector),
    the norm and absolute value, the normalized equivalent, and so on.  Also
    included are various

    The other main elements for a quaternionic array class are converters (to
    and from matrices, Euler angles, etc.)  and, of course, the algebraic
    functions (addition, multiplication, etc.).

    """

    @property
    def ndarray(self):
        return self.view(np.ndarray)

    @property
    @jit
    def flattened(self):
        return self.reshape((-1, 4))

    @property
    def w(self):
        return self.ndarray[..., 0]

    @w.setter
    def w(self, wprime):
        self.ndarray[..., 0] = wprime

    @property
    def x(self):
        return self.ndarray[..., 1]

    @x.setter
    def x(self, xprime):
        self.ndarray[..., 0] = xprime

    @property
    def y(self):
        return self.ndarray[..., 2]

    @y.setter
    def y(self, yprime):
        self.ndarray[..., 0] = yprime

    @property
    def z(self):
        return self.ndarray[..., 3]

    @z.setter
    def z(self, zprime):
        self.ndarray[..., 0] = zprime

    @property
    def vector(self):
        return self.ndarray[..., 1:]

    @vector.setter
    def vector(self, v):
        self.ndarray[..., 1:] = v

    @property
    @jit
    def norm(self):
        s = self.flattened
        n = np.empty(s.shape[0])
        for i in range(s.shape[0]):
            n[i] = s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2
        return n.reshape(self.shape[:-1])

    @property
    @jit
    def abs(self):
        s = self.flattened
        n = np.empty(s.shape[0])
        for i in range(s.shape[0]):
            n[i] = np.sqrt(s[i, 0]**2 + s[i, 1]**2 + s[i, 2]**2 + s[i, 3]**2)
        return n.reshape(self.shape[:-1])

    # Aliases
    scalar = w
    real = w
    i = x
    j = y
    k = z
    imag = vector
    absolute_square = norm
    squared_norm = norm
    abs2 = norm
    mag2 = norm
    modulus = abs
    magnitude = abs

    @property
    def normalized(self):
        return self / self.abs

    @property
    def iterator(self):
        s = self.flattened
        for i in range(s.shape[0]):
            yield(s[i])
