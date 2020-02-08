import numpy as np
from conjugate_gradient_method import solve
from test_triangulation import get_square_triang, get_square_circled_triang
from typing import List
from sparse_matrix import SparseMatrix
import matplotlib.pyplot as plt
import time

IS_LIQUID = True

ALPHA1 = 10  # air heat transfer coefficient     @unused
h_iz = 0.01  # height of isolation material    @unused
K_iz = 0.0883  # isolation material coefficient of thermal conductivity  @unused
ALPHA2 = ALPHA1 * K_iz / (K_iz + ALPHA1 * h_iz)  # isolation heat transfer coefficient   @unues
T_ENV = 20  # environment temperature           @ unused
T_DEF = 20  # defined border temperature
Q_DEF = 0  # heat flow                          @unused
Q_POINT = 5  # voltage of source points                      # 0
Kxx = -0.46  # main coefficient of thermal conductivity      $ 1
Kyy = -0.46  # main coefficient of thermal conductivity      $ 1

min_val = 0
max_val = 20

if IS_LIQUID:
    ALPHA1 = None
    h_iz = None
    K_iz = None
    ALPHA2 = None
    T_ENV = None
    T_DEF = None
    Q_DEF = None
    Q_POINT = 0
    Kxx = 1
    Kyy = 1


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


class Detail:
    border_points = None
    borders = None

    source_points = []  # type: List[Node]

    nodes = None
    elements = None

    isLiquid = False

    def __init__(self, isLiquid = False):  # customized for detail
        self.isLiquid = isLiquid
        quad_size = 100

        x_left = 0
        x_right = quad_size
        y_down = 0
        y_up = quad_size

        N = 40

        nodes, cells, border_nodes_indexes = get_square_circled_triang(N, quad_size) if isLiquid else get_square_triang(N, quad_size)

        # define detail border points
        self.border_points = [Node(index=1, x=x_left, y=y_down),
                       Node(index=2, x=x_left, y=y_up),
                       Node(index=3, x=x_right, y=y_up),
                       Node(index=4, x=x_right, y=y_down)]

        if not isLiquid:
            # define point sources
            self.source_points = [Node(index=0, x=(x_left + x_right) / 2, y=(y_down + y_up) / 2, q=Q_POINT)]

        # define detail borders
        self.borders = [(self.border_points[0], self.border_points[1]),
                        (self.border_points[1], self.border_points[2]),
                        (self.border_points[2], self.border_points[3]),
                        (self.border_points[3], self.border_points[0])]

        self.nodes = [Node(i, node[0], node[1]) for i, node in enumerate(nodes)]
        self.elements = [Element(i, self.nodes[cell[0]],
                                         self.nodes[cell[1]], self.nodes[cell[2]]) for i, cell in enumerate(cells)]

        if isLiquid:
            for i in border_nodes_indexes:
                self.nodes[i].t = 0

        self.define_border_conditions()

    def set_type_border(self, elem):
        """
        Description
        -----------
        If the element is on border defines its type: convective_heat_transfer, defined_T or heat_flow
        """
        for i in range(len(self.borders)):
            if self.isLiquid:
                if i in [0, 2]:
                    type = BorderType.HeatFlow
                    val = 0
                elif i in [1, 3]:
                    type = BorderType.DefinedTemperature
                    if i == 1:
                        val = max_val
                    else:
                        val = min_val
                else:
                    assert(False)
            else:
                if i == 0:
                    type = BorderType.ConvectiveHeatTransfer
                    val = ALPHA1
                elif i in [1, 3]:
                    type = BorderType.DefinedTemperature
                    val = T_DEF
                elif i == 2:
                    type = BorderType.HeatFlow
                    val = Q_DEF
                else:
                    assert(False)

            is_border = False

            if elem.s1.is_in_line(self.borders[i]) and elem.s2.is_in_line(self.borders[i]):
                is_border = True
                border = ElemBorderKey.first
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s1.index].t = val
                    self.nodes[elem.s2.index].t = val
            elif elem.s2.is_in_line(self.borders[i]) and elem.s3.is_in_line(self.borders[i]):
                is_border = True
                border = ElemBorderKey.second
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s2.index].t = val
                    self.nodes[elem.s3.index].t = val
            elif elem.s3.is_in_line(self.borders[i]) and elem.s1.is_in_line(self.borders[i]):
                is_border = True
                border = ElemBorderKey.third
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s3.index].t = val
                    self.nodes[elem.s1.index].t = val
            if is_border:
                elem.borders[border].type = type
                elem.borders[border].val = val

    def define_border_conditions(self):
        for elem in self.elements:
            self.set_type_border(elem)



class FEM:

    temps = None
    K = None
    F = None
    detail = None

    def __init__(self, detail):
        self.temps = []
        self.K = SparseMatrix(None, True)
        self.F = []
        self.detail = detail

    def build_system(self):
        """
        Description
        -----------
        Forms system of equations to solve
        """
        self.K.shape = (len(self.detail.nodes), len(self.detail.nodes))
        self.F = np.zeros(len(self.detail.nodes))
        for i, elem in enumerate(self.detail.elements):
            k = elem.form_elem_matrix(Kxx, Kyy)
            for j in range(3):
                for r in range(3):
                    self.K.add(index=(elem.s[j].index, elem.s[r].index), val=k[j][r])
            f = elem.form_vector_of_external_influences(self.detail.source_points)

            for j in range(3):
                self.F[elem.s[j].index] += f[j]

        for node in self.detail.nodes:
            if node.t is not None:
                self.K.set(index=(node.index, node.index), val=1)
                self.F[node.index] = node.t
                for node_k in self.detail.nodes:
                    if node_k.index != node.index:
                        self.K.set(index=(node.index, node_k.index), val=0)
                        self.F[node_k.index] -= self.K.get(index=(node_k.index, node.index)) * node.t
                        self.K.set(index=(node_k.index, node.index), val=0)

    def solve_system(self):
        self.temps = solve(self.K, self.F)

    def get_info(self):
        print('mesh: {} nodes, {} elements'.format(len(self.detail.nodes), len(self.detail.elements)))
        print('max temperature is {}'.format(np.max(self.temps)))
        print('min temperature is {}'.format(np.min(self.temps)))
        print('mean temperature is {}'.format(np.mean(self.temps)))

    def build_gradients(self):
        """
        Description
        -----------
        Builds gradients fields and view it
        """
        for elem in self.detail.elements:
            elem.grad = 1 / (2 * elem.A) * np.dot(
                np.array([[elem.b[0], elem.b[1], elem.b[2]], [elem.c[0], elem.c[1], elem.c[2]]]),
                np.array([[self.temps[elem.s1.index]], [self.temps[elem.s2.index]], [self.temps[elem.s3.index]]]))
        X = np.array([(elem.s1.x + elem.s2.x + elem.s3.x) / 3 for elem in self.detail.elements])
        Y = np.array([(elem.s1.y + elem.s2.y + elem.s3.y) / 3 for elem in self.detail.elements])
        U = np.array([elem.grad[0][0] for elem in self.detail.elements])
        V = np.array([elem.grad[1][0] for elem in self.detail.elements])

        fig3, ax3 = plt.subplots()
        speed = np.sqrt(U ** 2 + V ** 2)
        Q = ax3.quiver(X, Y, U, V, speed, width=0.0008)
        ax3.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                      coordinates='figure')
        ax3.scatter(X, Y, color='0.5', s=1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def create_vtu(self, file):
        """
        Description
        -----------
        Forms vtu file for paraview vizualization
        """
        output = '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" >\n\t<UnstructuredGrid>'
        output += '\n\t\t<Piece NumberOfPoints="{}" NumberOfCells="{}">'.format(len(self.detail.nodes), len(self.detail.elements))
        components = ''
        for node in self.detail.nodes:
            components += '{} {} 0 '.format(node.x, node.y)
        output += '\n\t\t<Points>\n\t\t\t<DataArray type="Float64" ' \
                  'NumberOfComponents="3" format="ascii">{}</DataArray>\n\t\t</Points>'.format(components)
        output += '\n\t\t<Cells>'
        connectivity = ''
        offsets = ''
        types = ''
        temps = ''

        for elem in self.detail.elements:
            connectivity += '{} {} {} '.format(elem.s1.index, elem.s2.index, elem.s3.index)
        for i in range(len(self.detail.elements)):
            offsets += '{} '.format((i + 1) * 3)
            types += '{} '.format(5)
        for t in self.temps:
            temps += '{} '.format(t)

        output += '\n\t\t\t<DataArray type="UInt32" Name="connectivity" format="ascii">{}</DataArray>'.format(
            connectivity)
        output += '\n\t\t\t<DataArray type="UInt32" Name="offsets" format="ascii">{}</DataArray>'.format(offsets)
        output += '\n\t\t\t<DataArray type="UInt8" Name="types" format="ascii">{}</DataArray>'.format(types)
        output += '\n\t\t</Cells>'
        output += '\n\t\t<PointData Scalars="T">\n\t\t\t<DataArray type="Float64" Name="T"' \
                  ' format="ascii">{}</DataArray>\n\t\t</PointData>'.format(temps)
        output += '\n\t\t</Piece>\n\t</UnstructuredGrid>\n</VTKFile>'

        f = open(file, "w+")
        f.write(output)
        f.close()


if __name__ == '__main__':
    start = time.time()

    fem = FEM(Detail(isLiquid=IS_LIQUID))

    fem.build_system()
    print('building system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.solve_system()
    print('solving system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.get_info()
    fem.build_gradients()
    fem.create_vtu('data.vtu')


