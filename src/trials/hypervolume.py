from pymoo.indicators.hv import HV

import numpy as np


A = np.array([[-1.2, 0.0], [-1, -0.25], [-0.25, -0.9], [0.0, -1.0]])
# A = np.array([[0.0, 1.0], [0.25, 0.75], [0.75, 0.25], [1.0, 0.0]])
B = np.array([[0.5, 0.2], [0.4, 0.3], [0.3, 0.4], [0.2, 0.5]])
# ref_pointA = np.array([0.0, 0.0])
ref_pointB = np.array([1.0, 0.5])

# indA = HV(ref_point=[0.0, 0.0])
indB = HV(ref_point=ref_pointB)

# print("HV", indA(A))
print("HV", indB(B))
