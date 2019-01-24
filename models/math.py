import numpy as np


class Vector:
    """ Vector: class of 3-dimensional vector for Coordinate and Direction """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get(self):
        return self.x, self.y, self.z

    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vectorize(self):
        return [self.x, self.y, self.z]

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def size(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    def projection(self, target):
        assert isinstance(target, Vector)
        return (self.dot(target) / target.size()) * (target / target.size())

    def scalar_projection(self, target):
        assert isinstance(target, Vector)
        return self.dot(target) / target.size()

    def get_cosine_angle(self, target):
        assert isinstance(target, Vector)
        return self.dot(target) / (self.size() * target.size())

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        return Vector(other.x + self.x, other.y + self.y, other.z + self.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y, other.z - self.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y and self.z == other.z


class Quaternion:
    """ Quaternion: class of 4-dimensional quaternion for Orientation and Rotation """
    def __init__(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def get(self):
        return self.w, self.i, self.j, self.k

    def update(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def vectorize(self):
        return [self.w, self.i, self.j, self.k]

    def get_vector_part(self):
        return Vector(self.i, self.j, self.k)

    def get_scalar_part(self):
        return self.w

    def get_conjugate(self):
        return Quaternion(self.w, -self.i, -self.j, -self.k)

    def is_unit(self):
        return np.square(self.w) + np.square(self.i) + np.square(self.j) + np.square(self.k) == 1.

    def __str__(self):
        return "(W:{w}, I:{i}, J:{j}, K:{k})".format(w=self.w, i=self.i, j=self.j, k=self.k)

    def __mul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=self.w*other.w - self.i*other.i - self.j*other.j - self.k*other.k,
            i=self.w*other.i + self.i*other.w + self.j*other.k - self.k*other.j,
            j=self.w*other.j - self.i*other.k + self.j*other.w + self.k*other.i,
            k=self.w*other.k + self.i*other.j - self.j*other.i + self.k*other.w
        )

    def __rmul__(self, other):
        assert isinstance(other, Quaternion)
        return Quaternion(
            w=other.w*self.w - other.i*self.i - other.j*self.j - other.k*self.k,
            i=other.w*self.i + other.i*self.w + other.j*self.k - other.k*self.j,
            j=other.w*self.j - other.i*self.k + other.j*self.w + other.k*self.i,
            k=other.w*self.k + other.i*self.j - other.j*self.i + other.k*self.w
        )
