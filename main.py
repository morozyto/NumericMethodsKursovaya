from conjugate_gradient_method import solve
from sparse_matrix import SparseMatrix
import matplotlib.pyplot as plt
import time
import math
from detail import *

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
        self.K.shape = (len(self.detail.nodes), len(self.detail.nodes))
        self.F = np.zeros(len(self.detail.nodes))
        for i, elem in enumerate(self.detail.elements):
            k = elem.form_elem_matrix(CONSTANTS.Kxx, CONSTANTS.Kyy)
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
        for elem in self.detail.elements:
            elem.grad = 1 / (2 * elem.A) * np.dot(
                np.array([[elem.b[0], elem.b[1], elem.b[2]], [elem.c[0], elem.c[1], elem.c[2]]]),
                np.array([[self.temps[elem.s1.index]], [self.temps[elem.s2.index]], [self.temps[elem.s3.index]]]))
        self.grad_X = np.array([(elem.s1.x + elem.s2.x + elem.s3.x) / 3 for elem in self.detail.elements])
        self.grad_Y = np.array([(elem.s1.y + elem.s2.y + elem.s3.y) / 3 for elem in self.detail.elements])
        self.grad_U = np.array([elem.grad[0][0] for elem in self.detail.elements])
        self.grad_V = np.array([elem.grad[1][0] for elem in self.detail.elements])

        fig3, ax3 = plt.subplots()
        speed = np.sqrt(self.grad_U ** 2 + self.grad_V ** 2)
        Q = ax3.quiver(self.grad_X, self.grad_Y, self.grad_U, self.grad_V, speed, width=0.0008)
        ax3.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                      coordinates='figure')
        ax3.scatter(self.grad_X, self.grad_Y, color='0.5', s=1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def create_vtu_vector(self, file):
        output = '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" >\n\t<UnstructuredGrid>'
        output += '\n\t\t<Piece NumberOfPoints="{}">'.format(len(self.detail.elements))
        components = ''
        for x, y in zip(self.grad_X, self.grad_Y):
            components += '{} {} 0 '.format(x, y)
        output += '\n\t\t<Points>\n\t\t\t<DataArray type="Float64" ' \
                  'NumberOfComponents="3" format="ascii">{}</DataArray>\n\t\t</Points>'.format(components)

        values = ''

        for u, v in zip(self.grad_U, self.grad_V):
            values += '{} '.format(math.sqrt(u*u + v*v))


        output += '\n\t\t<PointData Scalars="T">\n\t\t\t<DataArray type="Float64" Name="T"' \
                  ' format="ascii">{}</DataArray>\n\t\t</PointData>'.format(values)
        output += '\n\t\t</Piece>\n\t</UnstructuredGrid>\n</VTKFile>'

        f = open(file, "w+")
        f.write(output)
        f.close()

    def create_vtu(self, file):
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

    fem = FEM(Detail(isLiquid=CONSTANTS.IS_LIQUID))

    fem.build_system()
    print('building system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.solve_system()
    print('solving system in {} seconds'.format(time.time() - start))
    start = time.time()
    fem.get_info()
    fem.build_gradients()
    fem.create_vtu('data.vtu')
    fem.create_vtu_vector('gradient_data.vtu')

