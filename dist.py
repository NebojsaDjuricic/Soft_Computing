import math

# http://www.fundza.com/vectors/point2line/index.html


def distance(p0, p1):
    return length(vector(p0, p1))


def length(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return X - x, Y - y