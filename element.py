import numpy as np


class BorderType:
    HeatFlow = 'heat_flow'
    ConvectiveHeatTransfer = 'convective_heat_transfer'
    DefinedTemperature = 'defined_temperature'
    HeatIsolation = 'heat_isolation'
    NoBorder = 'no_border'

class ElemBorder:
    type = None
    val = None
    length = None
    vector = None

    def __init__(self, length, vector, type = BorderType.NoBorder, val=0):
        self.length = length
        self.vector = vector
        self.type = type
        self.val = val

class ElemBorderKey:
    first = '12'
    second = '23'
    third = '31'


class Node:
    index = None
    x = None
    y = None
    q = None
    t = None

    def __init__(self, index, x, y, q=0, t=None):
        self.index = index
        self.x = x
        self.y = y
        self.q = q
        self.t = t

    def is_in_line(self, line):
        return Node.sign(self, line[0], line[1]) == 0 and ((line[0].x - line[1].x != 0 and self.x > line[0].x and self.x < line[1].x)
                                                           or
                                                           (line[0].y - line[1].y != 0 and self.y > line[0].y and self.y < line[1].y))

    def sign(p1, p2, p3):
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)


class Element:

    index = None
    s1 = None
    s2 = None
    s3 = None
    A = None
    a = None
    b = None
    c = None
    borders = None

    def __init__(self, index, s1, s2, s3):
        self.index = index
        # define elem nodes
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

        # define elem area
        self.A = 0.5 * (self.s2.x * self.s3.y - self.s3.x * self.s2.y +
                        self.s1.x * self.s2.y - self.s1.x * self.s3.y +
                        self.s3.x * self.s1.y - self.s2.x * self.s1.y)

        # define additional arrays of coefs
        a_i = self.s2.x * self.s3.y - self.s3.x * self.s2.y
        a_j = self.s3.x * self.s1.y - self.s1.x * self.s3.y
        a_k = self.s1.x * self.s2.y - self.s2.x * self.s1.y

        b_i = self.s2.y - self.s3.y
        b_j = self.s3.y - self.s1.y
        b_k = self.s1.y - self.s2.y

        c_i = self.s3.x - self.s2.x
        c_j = self.s1.x - self.s3.x
        c_k = self.s2.x - self.s1.x

        self.a = [a_i, a_j, a_k]
        self.b = [b_i, b_j, b_k]
        self.c = [c_i, c_j, c_k]

        # define border length
        L12 = (self.c[2] ** 2 + self.b[2] ** 2) ** 0.5
        L23 = (self.c[0] ** 2 + self.b[0] ** 2) ** 0.5
        L31 = (self.c[1] ** 2 + self.b[1] ** 2) ** 0.5

        self.borders = {ElemBorderKey.first : ElemBorder(L12, np.array([1, 1, 0])),
                        ElemBorderKey.second: ElemBorder(L23, np.array([0, 1, 1])),
                        ElemBorderKey.third : ElemBorder(L31, np.array([1, 0, 1]))}

    @property
    def s(self):
        return [self.s1, self.s2, self.s3]

    def has_point_source(self, x0, y0):
        """
        Description
        -----------
        Check if element has heat point source or crosses it
        Parameters
        ----------
        x0: x coord of source
        y0: y coord of source
        """
        source = Node(None, x0, y0)
        d1 = Node.sign(source, self.s1, self.s2)
        d2 = Node.sign(source, self.s2, self.s3)
        d3 = Node.sign(source, self.s3, self.s1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def N(self, i, x, y):
        """
        Description
        -----------
        Returns value of basis function in point
        Parameters
        ----------
        i: index of basis function
        x: x coord of node
        y: y coord of node
        """
        return (1 / (2 * self.A)) * (self.a[i] + self.b[i] * x + self.c[i] * y)


    def form_elem_matrix(self, Kxx, Kyy):
        """
        Description
        -----------
        Forms left part of system
        Parameters
        ----------
        Kxx: thermal conductivity by x
        Kyy: thermal conductivity by y
        """
        k = (Kxx / (4 * self.A)) * np.array([[self.b[0] * self.b[0], self.b[0] * self.b[1], self.b[0] * self.b[2]],
                                             [self.b[1] * self.b[0], self.b[1] * self.b[1], self.b[1] * self.b[2]],
                                             [self.b[2] * self.b[0], self.b[2] * self.b[1], self.b[2] * self.b[2]]]) + \
            (Kyy / (4 * self.A)) * np.array([[self.c[0] * self.c[0], self.c[0] * self.c[1], self.c[0] * self.c[2]],
                                             [self.c[1] * self.c[0], self.c[1] * self.c[1], self.c[1] * self.c[2]],
                                             [self.c[2] * self.c[0], self.c[2] * self.c[1], self.c[2] * self.c[2]]])

        # check if border has convective_heat_transfer
        if self.borders[ElemBorderKey.first].type == BorderType.ConvectiveHeatTransfer:
            k += self.borders[ElemBorderKey.first].val * self.borders[ElemBorderKey.first].length / 6 * np.array(
                [[2, 1, 0], [1, 2, 0], [0, 0, 0]])
        if self.borders[ElemBorderKey.second].type == BorderType.ConvectiveHeatTransfer:
            k += self.borders[ElemBorderKey.second].val * self.borders[ElemBorderKey.second].length / 6 * np.array(
                [[0, 0, 0], [0, 2, 1], [0, 1, 2]])
        if self.borders[ElemBorderKey.third].type == BorderType.ConvectiveHeatTransfer:
            k += self.borders[ElemBorderKey.third].val * self.borders[ElemBorderKey.third].length / 6 * np.array(
                [[2, 0, 1], [0, 0, 0], [1, 0, 2]])
        return k

    def form_vector_of_external_influences(self, point_sources):
        """
        Description
        -----------
        Forms right part of system - vector of external influences
        Parameters
        ----------
        point_sources: array of point sources to check
        """
        f = np.zeros(3)
        for k in self.borders:
            if self.borders[k].val != 0:
                if self.borders[k].type == BorderType.HeatFlow:
                    f += self.borders[k].val / 2 * (self.borders[k].length * self.borders[k].vector)
                elif self.borders[k].type == BorderType.ConvectiveHeatTransfer:
                    f += self.borders[k].val * T_ENV / 2 * (self.borders[k].length * self.borders[k].vector)
        for p in point_sources:
            if self.has_point_source(p.x, p.y):
                f += p.q * np.array([self.N(0, p.x, p.y), self.N(1, p.x, p.y), self.N(2, p.x, p.y)])
        return f