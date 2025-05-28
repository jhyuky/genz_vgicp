import numpy as np

# 문자열 -> numpy array로 파싱하는 함수
def str_to_matrix(s):
    lines = s.strip().split('\n')
    return np.array([[float(num) for num in line.split()] for line in lines])

# Ground truth
# gt_str = """
# 0.999995946884 0.002785421442 0.000430414890 -0.049128323793
# -0.002782338765 0.999971508980 -0.007004556712 0.834670901299
# -0.000449913263 0.007003332023 0.999975383282 -0.086760424078
# 0.000000000000 0.000000000000 0.000000000000 1.000000000000
# """

# vgicp_str = """
#            1  0.000518842 -0.000654896    0.0238312
# -0.000518785            1  8.67544e-05   -0.0258672
#  0.000654941 -8.64146e-05            1    0.0162314
#            0            0            0            1
# """

# genz_str = """
#            1  0.000536915 -0.000660366    0.0125344
# -0.000536912            1   5.2912e-06  -0.00747344
#  0.000660369 -4.93662e-06            1    0.0148869
#            0            0            0            1
# """

# 복도 ground truth
gt_str = """
0.999845802784 -0.010070090182 -0.014392176643 0.315872490406
0.010524990037 0.999435901642 0.031889241189 -1.099993109703
0.014062931761 -0.032035805285 0.999387621880 -0.388353943825
0.000000000000 0.000000000000 0.000000000000 1.000000000000
"""

vgicp_str = """
   0.999451   0.0329563 -0.00336546    -1.41647
 -0.0329232    0.999413  0.00944882     2.47005
 0.00367487 -0.00933284     0.99995     2.23396
          0           0           0           1
"""
# 차이가 심할때
#    0.999451   0.0329563 -0.00336546    -1.41647
#  -0.0329232    0.999413  0.00944882     2.47005
#  0.00367487 -0.00933284     0.99995     2.23396
#           0           0           0           1


genz_str = """
    0.999973 -0.000141218  -0.00738439     0.123339
 0.000267892     0.999853    0.0171558     0.182585
  0.00738088   -0.0171574     0.999825    -0.224129
           0            0            0            1
"""
# 복도 ground truth

T_gt = str_to_matrix(gt_str)
T_vgicp = str_to_matrix(vgicp_str)
T_genz = str_to_matrix(genz_str)

def compute_errors(T_gt, T):
    # Translation Error
    Te = np.linalg.norm(T_gt[:3, 3] - T[:3, 3])
    # Rotation Error
    R_gt = T_gt[:3, :3]
    R = T[:3, :3]
    R_diff = R_gt @ np.linalg.inv(R)
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    Re = np.degrees(angle_rad)
    # Re = np.linalg.norm(R_diff - np.eye(3), ord='fro')  # Frobenius norm
    return Te, Re

# VGICP 결과
Te_vgicp, Re_vgicp = compute_errors(T_gt, T_vgicp)
print("VGICP: Te =", Te_vgicp, ", Re =", Re_vgicp)

# GenZ-VGICP 결과
Te_genz, Re_genz = compute_errors(T_gt, T_genz)
print("GenZ-VGICP: Te =", Te_genz, ", Re =", Re_genz)





# 병합이 잘되었을때
# --- VGICP ---
# RMSE: 1775.17
# Transform:
#     0.999972 -0.000762979  -0.00738993     0.170099
#    0.0008952     0.999839    0.0179052     0.118756
#   0.00737508   -0.0179113     0.999812    -0.188265
#            0            0            0            1
# Iterations: 49
# Converged: No
# Execution time: 77 ms
# Time per iteration: 1 ms

# --- Adaptive GenZ-VGICP ---
# RMSE: 1090.97
# Transform:
#     0.999973 -0.000141218  -0.00738439     0.123339
#  0.000267892     0.999853    0.0171558     0.182585
#   0.00738088   -0.0171574     0.999825    -0.224129
#            0            0            0            1
# Iterations: 15
# Converged: No
# Execution time: 35 ms

# 병합이 살짝 안맞을 때
# --- VGICP ---
# RMSE: 253.336
# Transform:
#    0.999451   0.0329563 -0.00336546    -1.41647
#  -0.0329232    0.999413  0.00944882     4.47005
#  0.00367487 -0.00933284     0.99995     2.23396
#           0           0           0           1
# Iterations: 29
# Converged: Yes
# Execution time: 47 ms
# Time per iteration: 1 ms

# --- Adaptive GenZ-VGICP ---
# RMSE: 238.952
# Transform:
#     0.99963   0.0271706  0.00129442   -0.784086
#  -0.0271428    0.999468  -0.0180736     1.63692
# -0.00178483   0.0180317    0.999836        1.65
#           0           0           0           1
# Iterations: 9
# Converged: No
# Execution time: 18 ms
# Time per iteration: 2 ms
