import numpy as np

class ChargeAsFastAsPossible:
    algo_name = "Charge As Fast As Possible"

    def __init__(self, verbose=False):
        self.verbose = verbose

    def get_action(self, env):
        return np.ones(env.number_of_ports)