SIDE_ELEMENTS_COUNT = 100

ALPHA1 = 10  # air heat transfer coefficient

h_iz = 0.01  # height of isolation material
K_iz = 0.0883  # isolation material coefficient of thermal conductivity
ALPHA2 = ALPHA1 * K_iz / (K_iz + ALPHA1 * h_iz)  # isolation heat transfer coefficient

T_ENV = 20  # environment temperature or flow value
T_DEF = 20  # defined border temperature
Q_DEF = 0  # heat flow
Q_POINT = 5  # voltage of source points

Kxx = -0.46  # main coefficient of thermal conductivity
Kyy = -0.46  # main coefficient of thermal conductivity

IS_LIQUID = False

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

    MIN_VAL = 0
    MAX_VAL = 20