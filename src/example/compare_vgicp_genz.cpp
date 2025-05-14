#include <iostream>
#include <Eigen/Geometry>

// 헤더
#include <small_gicp/ann/incremental_voxelmap.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/genz_vgicp_factor.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/reduction_omp.hpp>

// PCL 헤더
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

// Open3D 헤더
#include <open3d/Open3D.h>

// std::vector<Eigen::Vector4d>에 대한 traits 정의
namespace small_gicp {
namespace traits {
template <>
struct Traits<std::vector<Eigen::Vector4d>> {
  using Points = std::vector<Eigen::Vector4d>;
  
  static size_t size(const Points& points) { return points.size(); }
  static const Eigen::Vector4d& point(const Points& points, size_t i) { return points[i]; }
  static Eigen::Matrix4d cov(const Points& points, size_t i) { return Eigen::Matrix4d::Identity(); }
};

// PCL KdTree에 대한 traits 정의
template <>
struct Traits<pcl::KdTreeFLANN<pcl::PointXYZ>> {
  static size_t nearest_neighbor_search(
    const pcl::KdTreeFLANN<pcl::PointXYZ>& tree,
    const Eigen::Vector4d& point,
    size_t* k_index,
    double* k_sq_dist) {
    
    std::vector<int> indices(1);
    std::vector<float> sq_distances(1);
    
    pcl::PointXYZ pcl_point;
    pcl_point.x = point[0];
    pcl_point.y = point[1];
    pcl_point.z = point[2];
    
    if (tree.nearestKSearch(pcl_point, 1, indices, sq_distances) > 0) {
      *k_index = indices[0];
      *k_sq_dist = sq_distances[0];
      return 1;
    }
    return 0;
  }
};
}  // namespace traits
}  // namespace small_gicp

using namespace small_gicp;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " source.ply target.ply\n";
    return 1;
  }

  // 1) 포인트 클라우드 로드
  auto target_cloud_o3d = std::make_shared<open3d::geometry::PointCloud>();
  auto source_cloud_o3d = std::make_shared<open3d::geometry::PointCloud>();

  std::cout << "Loading target point cloud..." << std::endl;
  if (!open3d::io::ReadPointCloud(argv[2], *target_cloud_o3d)) {
    std::cerr << "Error: Failed to read target PLY file" << std::endl;
    return 1;
  }
  std::cout << "Target point cloud loaded. Points: " << target_cloud_o3d->points_.size() << std::endl;

  std::cout << "Loading source point cloud..." << std::endl;
  if (!open3d::io::ReadPointCloud(argv[1], *source_cloud_o3d)) {
    std::cerr << "Error: Failed to read source PLY file" << std::endl;
    return 1;
  }
  std::cout << "Source point cloud loaded. Points: " << source_cloud_o3d->points_.size() << std::endl;

  // 다운샘플링
  double voxel_size = 0.5;  // 50cm voxel size로 증가
  std::cout << "Downsampling point clouds with voxel size: " << voxel_size << std::endl;
  
  auto target_downsampled = target_cloud_o3d->VoxelDownSample(voxel_size);
  auto source_downsampled = source_cloud_o3d->VoxelDownSample(voxel_size);
  
  std::cout << "Downsampled target points: " << target_downsampled->points_.size() << std::endl;
  std::cout << "Downsampled source points: " << source_downsampled->points_.size() << std::endl;

  // 노이즈 제거
  target_downsampled = std::get<0>(target_downsampled->RemoveStatisticalOutliers(20, 2.0));
  source_downsampled = std::get<0>(source_downsampled->RemoveStatisticalOutliers(20, 2.0));

  std::cout << "After noise removal - Target points: " << target_downsampled->points_.size() << std::endl;
  std::cout << "After noise removal - Source points: " << source_downsampled->points_.size() << std::endl;

  // Open3D 포인트 클라우드를 Eigen::Vector4f로 변환
  std::vector<Eigen::Vector4f> target_points;
  std::vector<Eigen::Vector4f> source_points;
  
  target_points.reserve(target_downsampled->points_.size());
  for (const auto& pt : target_downsampled->points_) {
    target_points.emplace_back(pt.x(), pt.y(), pt.z(), 1.0f);
  }
  
  source_points.reserve(source_downsampled->points_.size());
  for (const auto& pt : source_downsampled->points_) {
    source_points.emplace_back(pt.x(), pt.y(), pt.z(), 1.0f);
  }
  
  if (target_points.empty() || source_points.empty()) {
    std::cerr << "Error: Failed to convert points" << std::endl;
    return 1;
  }

  // 원본 포인트 클라우드 메모리 해제
  target_cloud_o3d.reset();
  source_cloud_o3d.reset();
  target_downsampled.reset();
  source_downsampled.reset();

  // 2) 전처리 설정
  int num_threads = 4;
  double downsampling_resolution = 0.1;  // 더 작은 보셀 크기
  int num_neighbors = 20;  // 더 많은 이웃 포인트

  // 3) 포인트 클라우드 전처리
  auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
  auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

  // 4) VGICP 설정
  using VGICP = Registration<GICPFactor, ParallelReductionOMP>;
  VGICP reg_vg;
  reg_vg.point_factor.voxel_size = downsampling_resolution;
  reg_vg.reduction.num_threads = num_threads;
  reg_vg.criteria.rotation_eps = 0.1 * M_PI / 180.0;  // 회전 수렴 조건 완화
  reg_vg.criteria.translation_eps = 1e-3;  // 이동 수렴 조건 완화
  reg_vg.optimizer.max_iterations = 50;  // 최대 반복 횟수 감소

  // 5) GenZ-VGICP 설정
  using GZICP = Registration<GenZVGICPFactor, ParallelReductionOMP>;
  GZICP reg_gz;
  reg_gz.point_factor.voxel_size = downsampling_resolution;
  reg_gz.reduction.num_threads = num_threads;
  reg_gz.criteria.rotation_eps = 0.1 * M_PI / 180.0;  // 회전 수렴 조건 완화
  reg_gz.criteria.translation_eps = 1e-3;  // 이동 수렴 조건 완화
  reg_gz.optimizer.max_iterations = 50;  // 최대 반복 횟수 감소

  // 6) 초기 변환 설정 - 단순화된 초기화
  Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
  init.translation() = Eigen::Vector3d(-15.0, 0.0, 0.0);  // 단순화된 초기 위치
  init.linear() = Eigen::Matrix3d::Identity();  // 단순화된 초기 회전

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

  // 10) 정합된 source 포인트 클라우드를 변환하여 병합 결과를 PLY로 저장
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged_vgicp_cloud(new pcl::PointCloud<pcl::PointXYZ>());  // VGICP 결과용
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  // target 포인트 추가
  for (const auto& pt : target_points) {
    pcl::PointXYZ p;
    p.x = pt[0]; p.y = pt[1]; p.z = pt[2];
    merged_cloud->push_back(p);
    merged_vgicp_cloud->push_back(p);  // VGICP 결과에도 target 추가
    target_cloud->push_back(p);
  }

  // source 포인트를 변환하여 추가
  for (const auto& pt : source_points) {
    Eigen::Vector4d p = pt.cast<double>();
    
    // GenZ-VGICP 결과로 변환
    Eigen::Vector4d p_trans = res_gz.T_target_source * p;  // GenZ-VGICP 결과 사용
    pcl::PointXYZ pcl_p;
    pcl_p.x = p_trans[0];
    pcl_p.y = p_trans[1];
    pcl_p.z = p_trans[2];
    merged_cloud->push_back(pcl_p);

    // VGICP 결과로 변환
    Eigen::Vector4d p_trans_vgicp = res_vg.T_target_source * p;  // VGICP 결과 사용
    pcl::PointXYZ pcl_p_vgicp;
    pcl_p_vgicp.x = p_trans_vgicp[0];
    pcl_p_vgicp.y = p_trans_vgicp[1];
    pcl_p_vgicp.z = p_trans_vgicp[2];
    merged_vgicp_cloud->push_back(pcl_p_vgicp);

    // 원본 source 포인트 저장
    pcl::PointXYZ pcl_source;
    pcl_source.x = pt[0];
    pcl_source.y = pt[1];
    pcl_source.z = pt[2];
    source_cloud->push_back(pcl_source);
  }

  // PLY 파일로 저장
  pcl::io::savePLYFile("merged_genz_vgicp.ply", *merged_cloud);
  pcl::io::savePLYFile("merged_vgicp.ply", *merged_vgicp_cloud);
  pcl::io::savePLYFile("target.ply", *target_cloud);
  pcl::io::savePLYFile("source.ply", *source_cloud);
  std::cout << "\n병합된 결과를 merged_genz_vgicp.ply로 저장했습니다." << std::endl;
  std::cout << "VGICP 병합 결과를 merged_vgicp.ply로 저장했습니다." << std::endl;
  std::cout << "target 포인트 클라우드를 target.ply로 저장했습니다." << std::endl;
  std::cout << "source 포인트 클라우드를 source.ply로 저장했습니다." << std::endl;

  return 0;
}