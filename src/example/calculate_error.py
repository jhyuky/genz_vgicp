import numpy as np

# 문자열 -> numpy array로 파싱하는 함수
def str_to_matrix(s):
    lines = s.strip().split('\n')
    return np.array([[float(num) for num in line.split()] for line in lines])

# Ground truth
gt_str = """
0.999969542027 0.007775768638 0.000689222186 -0.096736900508
-0.007774004713 0.999966561794 -0.002526176628 0.763605117798
-0.000708842126 0.002520741662 0.999996542931 -0.108479268849
0.000000000000 0.000000000000 0.000000000000 1.000000000000
"""

vgicp_str = """
1  0.000518724 -0.000654233    0.0137068
-0.000518664 1   8.7266e-05   -0.0159139
0.000654278 -8.69266e-05 1    0.0162657
0 0 0 1
"""

genz_str = """
1  0.000578252 -0.000371964  -0.00460665
-0.000578355 1  -0.00028331    0.0197307
0.0003718  0.000283525 1    0.0353066
0 0 0 1
"""

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
    Re = np.linalg.norm(R_diff - np.eye(3), ord='fro')  # Frobenius norm
    return Te, Re

# VGICP 결과
Te_vgicp, Re_vgicp = compute_errors(T_gt, T_vgicp)
print("VGICP: Te =", Te_vgicp, ", Re =", Re_vgicp)

# GenZ-VGICP 결과
Te_genz, Re_genz = compute_errors(T_gt, T_genz)
print("GenZ-VGICP: Te =", Te_genz, ", Re =", Re_genz)
