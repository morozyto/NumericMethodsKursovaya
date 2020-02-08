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