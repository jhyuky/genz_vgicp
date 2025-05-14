#include <iostream>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/genz_vgicp_factor.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>

using namespace small_gicp;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " source.ply target.ply\n";
    return 1;
  }

  // 1) 포인트 클라우드 로드
  std::vector<Eigen::Vector4f> target_points = read_ply(argv[2]);
  std::vector<Eigen::Vector4f> source_points = read_ply(argv[1]);
  
  if (target_points.empty() || source_points.empty()) {
    std::cerr << "Error: Failed to read points from PLY files" << std::endl;
    return 1;
  }

  // 2) 전처리 설정
  int num_threads = 4;
  double downsampling_resolution = 0.25;
  int num_neighbors = 10;

  // 3) 포인트 클라우드 전처리
  auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
  auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

  // 4) VGICP 설정
  using VGICP = Registration<GICPFactor, ParallelReductionOMP>;
  VGICP reg_vg;
  reg_vg.point_factor.voxel_size = downsampling_resolution;
  reg_vg.reduction.num_threads = num_threads;

  // 5) GenZ-VGICP 설정
  using GZICP = Registration<GenZVGICPFactor, ParallelReductionOMP>;
  GZICP reg_gz;
  reg_gz.point_factor.voxel_size = downsampling_resolution;
  reg_gz.reduction.num_threads = num_threads;

  // 6) 초기 변환 설정
  Eigen::Isometry3d init = Eigen::Isometry3d::Identity();

  // 7) 정합 실행
  auto res_vg = reg_vg.align(*target, *source, *target_tree, init);
  auto res_gz = reg_gz.align(*target, *source, *target_tree, init);

  // 8) 결과 출력
  std::cout << "\n--- VGICP ---\n"
            << "RMSE: " << res_vg.error << "\n"
            << "Transform:\n" << res_vg.T_target_source.matrix() << "\n"
            << "Iterations: " << res_vg.iterations << "\n"
            << "Converged: " << (res_vg.converged ? "Yes" : "No") << "\n";

  std::cout << "\n--- GenZ-VGICP ---\n"
            << "RMSE: " << res_gz.error << "\n"
            << "Transform:\n" << res_gz.T_target_source.matrix() << "\n"
            << "Iterations: " << res_gz.iterations << "\n"
            << "Converged: " << (res_gz.converged ? "Yes" : "No") << "\n";

  // 9) 두 변환의 차이 계산
  Eigen::Isometry3d delta = res_gz.T_target_source * res_vg.T_target_source.inverse();
  std::cout << "\n--- 변환 차이 ---\n" << delta.matrix() << "\n";

  return 0;
}