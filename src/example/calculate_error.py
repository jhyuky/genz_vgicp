import numpy as np

# 문자열 -> numpy array로 파싱하는 함수
def str_to_matrix(s):
    lines = s.strip().split('\n')
    return np.array([[float(num) for num in line.split()] for line in lines])

# Ground truth
gt_str = """
0.999999344349 0.001154619269 -0.000132486559 0.054880205542
-0.001154665370 0.999999284744 -0.000348161178 0.045637980103
0.000132084475 0.000348313915 0.999999940395 0.008304588497
0.000000000000 0.000000000000 0.000000000000 1.000000000000
"""

vgicp_str = """
    0.999999  0.000611591  -0.00120183     -118.971
-0.000614983     0.999996   -0.0028245      3.22419
   0.0012001   0.00282524     0.999995      66.2212
           0            0            0            1
"""

genz_str = """
    0.999999   0.00059027  -0.00131419     -119.058
-0.000593821     0.999996  -0.00270356      3.18884
  0.00131259   0.00270434     0.999995      66.2089
           0            0            0            1
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
