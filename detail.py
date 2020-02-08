from details_triangulations import get_square_triang, get_square_circled_triang
from element import *

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

        N = CONSTANTS.SIDE_ELEMENTS_COUNT

        nodes, cells, border_nodes_indexes = get_square_circled_triang(N, quad_size) if isLiquid else get_square_triang(N, quad_size)

        # define detail border points
        self.border_points = [Node(index=1, x=x_left, y=y_down),
                       Node(index=2, x=x_left, y=y_up),
                       Node(index=3, x=x_right, y=y_up),
                       Node(index=4, x=x_right, y=y_down)]

        if not isLiquid:
            # define point sources
            self.source_points = [Node(index=0, x=(x_left + x_right) / 2, y=(y_down + y_up) / 2, q=CONSTANTS.Q_POINT)]

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
                        val = CONSTANTS.MAX_VAL
                    else:
                        val = CONSTANTS.MIN_VAL
                else:
                    assert(False)
            else:
                if i == 0:
                    type = BorderType.ConvectiveHeatTransfer
                    val = CONSTANTS.ALPHA1
                elif i in [1, 3]:
                    type = BorderType.DefinedTemperature
                    val = CONSTANTS.T_DEF
                elif i == 2:
                    type = BorderType.HeatFlow
                    val = CONSTANTS.Q_DEF
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
