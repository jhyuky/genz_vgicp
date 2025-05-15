#!/usr/bin/python3
import open3d as o3d
import numpy as np
import sys
import os

def visualize_ply_files(target_file, source_file, merged_genz_file, merged_vgicp_file):
    # PLY 파일들 읽기
    print("PLY 파일 로드 중...")
    target_pcd = o3d.io.read_point_cloud(target_file)
    source_pcd = o3d.io.read_point_cloud(source_file)
    merged_genz_pcd = o3d.io.read_point_cloud(merged_genz_file)
    merged_vgicp_pcd = o3d.io.read_point_cloud(merged_vgicp_file)
    
    if not target_pcd.has_points() or not source_pcd.has_points() or not merged_genz_pcd.has_points() or not merged_vgicp_pcd.has_points():
        print("오류: 하나 이상의 포인트 클라우드가 비어있습니다.")
        return
    
    # 포인트 클라우드 위치 조정
    target_center = target_pcd.get_center()
    source_center = source_pcd.get_center()
    
    # source 포인트 클라우드를 target 기준으로 오프셋 적용
    source_pcd.translate([5.0, 0, 0])  # x축으로 5.0 이동
    
    # 각 포인트 클라우드에 색상 지정 (더 진한 색상으로 변경)
    target_pcd.paint_uniform_color([0.8, 0.2, 0.2])  # 진한 빨간색
    source_pcd.paint_uniform_color([0.2, 0.8, 0.2])  # 진한 초록색
    merged_genz_pcd.paint_uniform_color([0.2, 0.2, 0.8])  # 진한 파란색
    merged_vgicp_pcd.paint_uniform_color([0.8, 0.2, 0.8])  # 진한 보라색
    
    # 포인트 크기 설정
    target_pcd.estimate_normals()
    source_pcd.estimate_normals()
    merged_genz_pcd.estimate_normals()
    merged_vgicp_pcd.estimate_normals()
    
    # 포인트 클라우드 정보 출력
    print(f"\n포인트 클라우드 정보:")
    print(f"Target 포인트 수: {len(target_pcd.points)}")
    print(f"Source 포인트 수: {len(source_pcd.points)}")
    print(f"GenZ-VGICP 병합 포인트 수: {len(merged_genz_pcd.points)}")
    print(f"VGICP 병합 포인트 수: {len(merged_vgicp_pcd.points)}")
    
    # Target과 Source를 위한 시각화 창
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Target & Source", width=1280, height=720)
    
    # GenZ-VGICP 결과를 위한 시각화 창
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="GenZ-VGICP Result", width=1280, height=720)
    
    # VGICP 결과를 위한 시각화 창
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name="VGICP Result", width=1280, height=720)
    
    # Target과 Source 포인트 클라우드 추가
    vis1.add_geometry(target_pcd)
    vis1.add_geometry(source_pcd)
    
    # GenZ-VGICP 결과 추가
    vis2.add_geometry(merged_genz_pcd)
    
    # VGICP 결과 추가
    vis3.add_geometry(merged_vgicp_pcd)
    
    # 렌더링 옵션 설정 (Target & Source)
    opt1 = vis1.get_render_option()
    opt1.point_size = 2.0  # 포인트 크기 조정
    opt1.background_color = np.asarray([1, 1, 1])  # 흰색 배경
    opt1.light_on = True
    opt1.line_width = 2.0  # 라인 두께 증가
    
    # 렌더링 옵션 설정 (GenZ-VGICP)
    opt2 = vis2.get_render_option()
    opt2.point_size = 2.0  # 포인트 크기 조정
    opt2.background_color = np.asarray([1, 1, 1])  # 흰색 배경
    opt2.light_on = True
    opt2.line_width = 2.0  # 라인 두께 증가
    
    # 렌더링 옵션 설정 (VGICP)
    opt3 = vis3.get_render_option()
    opt3.point_size = 2.0  # 포인트 크기 조정
    opt3.background_color = np.asarray([1, 1, 1])  # 흰색 배경
    opt3.light_on = True
    opt3.line_width = 2.0  # 라인 두께 증가
    
    # 카메라 위치 설정 (Target & Source)
    ctr1 = vis1.get_view_control()
    ctr1.set_zoom(0.5)  # 줌 레벨 조정
    ctr1.set_front([0, 0, -1])
    ctr1.set_lookat([0, 0, 0])
    ctr1.set_up([0, -1, 0])
    
    # 카메라 위치 설정 (GenZ-VGICP)
    ctr2 = vis2.get_view_control()
    ctr2.set_zoom(0.5)  # 줌 레벨 조정
    ctr2.set_front([0, 0, -1])
    ctr2.set_lookat([0, 0, 0])
    ctr2.set_up([0, -1, 0])
    
    # 카메라 위치 설정 (VGICP)
    ctr3 = vis3.get_view_control()
    ctr3.set_zoom(0.5)  # 줌 레벨 조정
    ctr3.set_front([0, 0, -1])
    ctr3.set_lookat([0, 0, 0])
    ctr3.set_up([0, -1, 0])
    
    print("\n시각화 창을 닫으려면 'q'를 누르세요.")
    print("Target & Source 창:")
    print("빨간색: Target 포인트 클라우드 (중앙)")
    print("초록색: Source 포인트 클라우드 (오른쪽)")
    print("\nGenZ-VGICP Result 창:")
    print("파란색: GenZ-VGICP 병합 포인트 클라우드")
    print("\nVGICP Result 창:")
    print("보라색: VGICP 병합 포인트 클라우드")
    
    # 세 창을 동시에 실행
    while True:
        vis1.update_geometry(target_pcd)
        vis1.update_geometry(source_pcd)
        vis1.poll_events()
        vis1.update_renderer()
        
        vis2.update_geometry(merged_genz_pcd)
        vis2.poll_events()
        vis2.update_renderer()
        
        vis3.update_geometry(merged_vgicp_pcd)
        vis3.poll_events()
        vis3.update_renderer()
        
        if not vis1.poll_events() or not vis2.poll_events() or not vis3.poll_events():
            break
    
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()

def main():
    # PLY 파일 경로 설정
    # target_file = "/home/nvidia/paper_ws/src/small_gicp/target.ply"
    # source_file = "/home/nvidia/paper_ws/src/small_gicp/source.ply"
    # merged_genz_file = "/home/nvidia/paper_ws/src/small_gicp/merged_genz_vgicp.ply"
    # merged_vgicp_file = "/home/nvidia/paper_ws/src/small_gicp/merged_vgicp.ply"

    target_file = "/home/nvidia/paper_ws/src/small_gicp/target_offset.ply"
    source_file = "/home/nvidia/paper_ws/src/small_gicp/source_offset.ply"
    merged_genz_file = "/home/nvidia/paper_ws/src/small_gicp/merged_genz_vgicp_offset.ply"
    merged_vgicp_file = "/home/nvidia/paper_ws/src/small_gicp/merged_vgicp_offset.ply"
    
    visualize_ply_files(target_file, source_file, merged_genz_file, merged_vgicp_file)

if __name__ == "__main__":
    main() 