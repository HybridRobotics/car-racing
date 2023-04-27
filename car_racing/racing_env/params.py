X_DIM, U_DIM = 6, 2

class BicycleDynamicsParam:
    def __init__(
        self,
        m=1.98,
        lf=0.125,
        lr=0.125,
        Iz=0.024,
        Df=0.8 * 1.98 * 9.81 / 2.0,
        Cf=1.25,
        Bf=1.0,
        Dr=0.8 * 1.98 * 9.81 / 2.0,
        Cr=1.25,
        Br=1.0,
    ):
        self.m = m
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.Df = Df
        self.Cf = Cf
        self.Bf = Bf
        self.Dr = Dr
        self.Cr = Cr
        self.Br = Br

    def get_params(self):
        return (
            self.m,
            self.lf,
            self.lr,
            self.Iz,
            self.Df,
            self.Cf,
            self.Bf,
            self.Dr,
            self.Cr,
            self.Br,
        )

class SystemParam:
    def __init__(self, delta_max=0.5, a_max=1.0, v_max=10, v_min=0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min

class CarParam:
    def __init__(self, length=0.4, width=0.2, facecolor="None", edgecolor="black"):
        self.length = length
        self.width = width
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dynamics_param = BicycleDynamicsParam()

