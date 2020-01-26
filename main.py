import numpy as np
from conjugate_gradient_method import solve
from typing import List
from sparse_matrix import SparseMatrix
import matplotlib.pyplot as plt
import time

ALPHA1 = 10  # air heat transfer coefficient
h_iz = 0.01  # height of isolation material
K_iz = 0.0883  # isolation material coefficient of thermal conductivity
ALPHA2 = ALPHA1 * K_iz / (K_iz + ALPHA1 * h_iz)  # isolation heat transfer coefficient
T_ENV = 20  # environment temperature
T_DEF = 20  # defined border temperature
Q_DEF = 0  # heat flow
Q_POINT = 5  # voltage of source points
Kxx = -0.46  # main coefficient of thermal conductivity
Kyy = -0.46  # main coefficient of thermal conductivity



class BorderType:
    HeatFlow = 'heat_flow'
    ConvectiveHeatTransfer = 'convective_heat_transfer'
    DefinedTemperature = 'defined_temperature'
    HeatIsolation = 'heat_isolation'
    NoBorder = 'no_border'



class Detail:
    source_points = None  # type: List[Node]
    nodes = []
    elements = []

    def __init__(self):  # customized for detail

        x_left = 0
        x_right = 100
        y_down = 0
        y_up = 100

        nodes, cells = get_square_triang(10, 100)

        # define detail border points
        self.border_points = [Node(index=1, x=x_left, y=y_down),
                       Node(index=2, x=x_left, y=y_up),
                       Node(index=3, x=x_right, y=y_up),
                       Node(index=4, x=x_right, y=y_down)]

        # define point sources
        self.source_points = [Node(index=0, x=(x_left + x_right) / 2, y=(y_left + y_right) / 2, q=Q_POINT)]

        # define detail borders
        self.borders = [(self.border_points[0], self.border_points[1]),
                        (self.border_points[1], self.border_points[2]),
                        (self.border_points[2], self.border_points[3]),
                        (self.border_points[3], self.border_points[0])]

        self.nodes = [Node(i, node[0], node[1]) for i, node in enumerate(nodes)]








class Node:
    def __init__(self, index, x, y, q=0, t=None):
        self.index = index
        self.x = x
        self.y = y
        self.q = q
        self.t = t

    def is_in_line(self, line):
        return sign(self, line[0], line[1]) == 0

    def sign(p1, p2, p3):
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)









class Element:
    def __init__(self, index, s1, s2, s3):
        self.index = index
        # define elem nodes
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s = [self.s1, self.s2, self.s3]

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
        self.borders = {'12': {'type': BorderType.NoBorder, 'val': 0, 'length': L12, 'vector': np.array([1, 1, 0])},
                        '23': {'type': BorderType.NoBorder, 'val': 0, 'length': L23, 'vector': np.array([0, 1, 1])},
                        '31': {'type': BorderType.NoBorder, 'val': 0, 'length': L31, 'vector': np.array([1, 0, 1])}}




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
        d1 = sign(source, self.s1, self.s2)
        d2 = sign(source, self.s2, self.s3)
        d3 = sign(source, self.s3, self.s1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        is_in = d1 == 0 or d2 == 0 or d3 == 0

        return not (has_neg and has_pos) or is_in

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
        if self.borders['12']['type'] == BorderType.ConvectiveHeatTransfer:
            k += self.borders['12']['val'] * self.borders['12']['length'] / 6 * np.array(
                [[2, 1, 0], [1, 2, 0], [0, 0, 0]])
        if self.borders['23']['type'] == BorderType.ConvectiveHeatTransfer:
            k += self.borders['23']['val'] * self.borders['23']['length'] / 6 * np.array(
                [[0, 0, 0], [0, 2, 1], [0, 1, 2]])
        if self.borders['31']['type'] == BorderType.ConvectiveHeatTransfer:
            k += self.borders['31']['val'] * self.borders['31']['length'] / 6 * np.array(
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
            if self.borders[k]['val'] != 0:
                if self.borders[k]['type'] == BorderType.HeatFlow:
                    f += self.borders[k]['val'] / 2 * (self.borders[k]['length'] * self.borders[k]['vector'])
                elif self.borders[k]['type'] == BorderType.ConvectiveHeatTransfer:
                    f += self.borders[k]['val'] * T_ENV / 2 * (self.borders[k]['length'] * self.borders[k]['vector'])
        for p in point_sources:
            if self.has_point_source(p.x, p.y):
                f += p.q * np.array([self.N(0, p.x, p.y), self.N(1, p.x, p.y), self.N(2, p.x, p.y)])
        return f



class FEM:
    def __init__(self, detail):
        self.nodes = [ Node(int(vertex.get('index')), float(vertex.get('x')), float(vertex.get('y'))) for vertex in vertices.getchildren()  ]
        self.elements = [ Element(int(cell.get('index')), self.nodes[int(cell.get('v0'))],
                                         self.nodes[int(cell.get('v1'))], self.nodes[int(cell.get('v2'))]) for cell in cells.getchildren() ]
        self.temps = []
        self.K = SparseMatrix(None, True)
        self.F = []
        self.detail = detail



    def set_type_border(self, elem):
        """
        Description
        -----------
        If the element is on border defines its type: convective_heat_transfer, defined_T or heat_flow
        """
        is_border = False
        for i in range(len(self.detail.borders)):
            if i in [0, 1, 2, 3, 5]:
                type = BorderType.ConvectiveHeatTransfer
                if i in [3, 5]:
                    val = ALPHA1
                else:
                    val = ALPHA2
            elif i == 4:
                type = BorderType.DefinedTemperature
                val = T_DEF
            else:
                type = BorderType.HeatFlow
                val = Q_DEF

            if elem.s1.is_in_line(self.detail.borders[i]) and elem.s2.is_in_line(self.detail.borders[i]):
                is_border = True
                border = '12'
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s1.index].t = val
                    self.nodes[elem.s2.index].t = val
            if elem.s2.is_in_line(self.detail.borders[i]) and elem.s3.is_in_line(self.detail.borders[i]):
                is_border = True
                border = '23'
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s2.index].t = val
                    self.nodes[elem.s3.index].t = val
            if elem.s3.is_in_line(self.detail.borders[i]) and elem.s1.is_in_line(self.detail.borders[i]):
                is_border = True
                border = '31'
                if type == BorderType.DefinedTemperature:
                    self.nodes[elem.s3.index].t = val
                    self.nodes[elem.s1.index].t = val
            if is_border:
                elem.borders[border]['type'] = type
                elem.borders[border]['val'] = val
            is_border = False

    def define_border_conditions(self):
        for elem in self.elements:
            self.set_type_border(elem)

    def build_system(self):
        """
        Description
        -----------
        Forms system of equations to solve
        """
        self.K.shape = (len(self.nodes), len(self.nodes))
        self.F = np.zeros(len(self.nodes))
        for i, elem in enumerate(self.elements):
            k = elem.form_elem_matrix(Kxx, Kyy)
            for j in range(3):
                for r in range(3):
                    self.K.add(index=(elem.s[j].index, elem.s[r].index), val=k[j][r])
            f = elem.form_vector_of_external_influences(self.detail.source_points)

            for j in range(3):
                self.F[elem.s[j].index] += f[j]

        for node in self.nodes:
            if node.t is not None:
                self.K.set(index=(node.index, node.index), val=1)
                self.F[node.index] = node.t
                for node_k in self.nodes:
                    if node_k.index != node.index:
                        self.K.set(index=(node.index, node_k.index), val=0)
                        self.F[node_k.index] -= self.K.get(index=(node_k.index, node.index)) * node.t
                        self.K.set(index=(node_k.index, node.index), val=0)

    def solve_system(self):
        self.temps = solve(self.K, self.F)

    def get_info(self):
        print('mesh: {} nodes, {} elements'.format(len(self.nodes), len(self.elements)))
        print('max temperature is {}'.format(np.max(self.temps)))
        print('min temperature is {}'.format(np.min(self.temps)))
        print('mean temperature is {}'.format(np.mean(self.temps)))

    def build_gradients(self):
        """
        Description
        -----------
        Builds gradients fields and view it
        """
        for elem in self.elements:
            for p in self.detail.source_points:
                if abs(p.x - elem.s1.x) < 0.0001 and abs(p.y - elem.s1.y) < 0.0001:
                    elem.ps = True
                    break
                else:
                    elem.ps = False
            elem.grad = 1 / (2 * elem.A) * np.dot(
                np.array([[elem.b[0], elem.b[1], elem.b[2]], [elem.c[0], elem.c[1], elem.c[2]]]),
                np.array([[self.temps[elem.s1.index]], [self.temps[elem.s2.index]], [self.temps[elem.s3.index]]]))
        w = 3
        X = np.array([(elem.s1.x + elem.s2.x + elem.s3.x) / 2 for elem in self.elements])
        Y = np.array([(elem.s1.y + elem.s2.y + elem.s3.y) / 2 for elem in self.elements])
        U = np.array([elem.grad[0][0] for elem in self.elements])
        V = np.array([elem.grad[1][0] for elem in self.elements])

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
        output += '\n\t\t<Piece NumberOfPoints="{}" NumberOfCells="{}">'.format(len(self.nodes), len(self.elements))
        components = ''
        for node in self.nodes:
            components += '{} {} 0 '.format(node.x, node.y)
        output += '\n\t\t<Points>\n\t\t\t<DataArray type="Float64" ' \
                  'NumberOfComponents="3" format="ascii">{}</DataArray>\n\t\t</Points>'.format(components)
        output += '\n\t\t<Cells>'
        connectivity = ''
        offsets = ''
        types = ''
        temps = ''

        for elem in self.elements:
            connectivity += '{} {} {} '.format(elem.s1.index, elem.s2.index, elem.s3.index)
        for i in range(len(self.elements)):
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
    fem = FEM(Detail())
    print('reading mesh in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.define_border_conditions()
    print('defining border conditions in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.build_system()
    print('building system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.solve_system()
    print('solving system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.get_info()
    fem.build_gradients()
    fem.create_vtu('mesh/data.vtu')

