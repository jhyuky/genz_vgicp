#include <iostream>
#include <Eigen/Geometry>
#include <chrono>  // 시간 측정용 헤더 추가

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

// PointT 타입 정의
using PointT = pcl::PointXYZ;

// VGICP 설정 구조체 정의
struct VGICPSetting {
  double voxel_size;
  int num_threads;
  double rotation_eps;
  double translation_eps;
  int max_iterations;
};

// GaussianVoxelMap 생성 함수
template <typename PointCloud>
std::shared_ptr<GaussianVoxelMap> create_gaussian_voxelmap(const PointCloud& points, double voxel_size) {
  auto voxelmap = std::make_shared<GaussianVoxelMap>(voxel_size);
  voxelmap->insert(points);
  return voxelmap;
}

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

  // Open3D 포인트 클라우드를 Eigen::Vector4d로 변환
  std::vector<Eigen::Vector4d> target_points;
  std::vector<Eigen::Vector4d> source_points;
  
  target_points.reserve(target_downsampled->points_.size());
  for (const auto& pt : target_downsampled->points_) {
    target_points.emplace_back(pt.x(), pt.y(), pt.z(), 1.0);
  }
  
  source_points.reserve(source_downsampled->points_.size());
  for (const auto& pt : source_downsampled->points_) {
    source_points.emplace_back(pt.x(), pt.y(), pt.z(), 1.0);
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
  double downsampling_resolution = 0.5;  // 보셀 크기 증가
  int num_neighbors = 20;  // 더 많은 이웃 포인트

  // 3) 포인트 클라우드 전처리
  auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
  auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

  // 4) VGICP 설정
  small_gicp::RegistrationSetting vgicp_setting;
  vgicp_setting.type = small_gicp::RegistrationSetting::VGICP;
  vgicp_setting.voxel_resolution = downsampling_resolution;
  vgicp_setting.max_correspondence_distance = 2.0 * downsampling_resolution;
  vgicp_setting.num_threads = num_threads;
  vgicp_setting.rotation_eps = 0.1 * M_PI / 180.0;
  vgicp_setting.translation_eps = 1e-3;
  vgicp_setting.max_iterations = 50;
  vgicp_setting.verbose = true;

  // 5) GenZ-VGICP 설정
  GenZVGICPFactor::Setting genz_factor_setting;
  genz_factor_setting.voxel_size = voxel_size;
  genz_factor_setting.alpha_v = 0.2;
  genz_factor_setting.alpha_g = 0.8;
  genz_factor_setting.error_scale = 1.0;
  genz_factor_setting.use_combined_cov = true;

  Registration<GenZVGICPFactor, ParallelReductionOMP> genz_vgicp;
  genz_vgicp.point_factor = genz_factor_setting;
  genz_vgicp.reduction.num_threads = num_threads;
  genz_vgicp.criteria.rotation_eps = 1.0 * M_PI / 180.0;
  genz_vgicp.criteria.translation_eps = 1e-2;
  genz_vgicp.optimizer.max_iterations = 50;
  genz_vgicp.optimizer.verbose = true;
  genz_vgicp.rejector.max_dist_sq = 5.0 * voxel_size * voxel_size;

  // 6) 초기 변환 설정
  Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
  
  // // 포인트 클라우드의 중심 계산
  // Eigen::Vector3d target_center = Eigen::Vector3d::Zero();
  // Eigen::Vector3d source_center = Eigen::Vector3d::Zero();
  
  // for (const auto& pt : target_points) {
  //   target_center += pt.head<3>().cast<double>();
  // }
  // target_center /= target_points.size();
  
  // for (const auto& pt : source_points) {
  //   source_center += pt.head<3>().cast<double>();
  // }
  // source_center /= source_points.size();
  
  // // 초기 변환 설정
  // init.translation() = target_center - source_center;
  // init.linear() = Eigen::Matrix3d::Identity();

  // Teaser ++ 사용한 초기 변환 행렬 설정
  Eigen::Matrix4d init_matrix;
  init_matrix << 0.998769760132, 0.044464513659, 0.021951656789, -2.019068956375,
                -0.044545717537, 0.999002158642, 0.003224032931, 2.398455381393,
                -0.021786397323, -0.004197918810, 0.999753832817, 0.923875689507,
                0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;
  
  init.matrix() = init_matrix;

  // 7) 정합 실행
  std::cout << "Running VGICP..." << std::endl;
  // GaussianVoxelMap 생성 및 설정
  auto gmap = create_gaussian_voxelmap(*target, downsampling_resolution);
  gmap->set_search_offsets(27);  // VGICP의 기본값 사용

  // VGICP 시간 측정 시작
  auto vgicp_start = std::chrono::high_resolution_clock::now();
  
  // VGICP align 호출
  using VGICP = Registration<GICPFactor, ParallelReductionOMP>;
  VGICP vgicp;
  vgicp.reduction.num_threads = vgicp_setting.num_threads;
  vgicp.criteria.rotation_eps = vgicp_setting.rotation_eps;
  vgicp.criteria.translation_eps = vgicp_setting.translation_eps;
  vgicp.optimizer.max_iterations = vgicp_setting.max_iterations;
  vgicp.optimizer.verbose = vgicp_setting.verbose;
  vgicp.rejector.max_dist_sq = vgicp_setting.max_correspondence_distance * vgicp_setting.max_correspondence_distance;
  
  auto res_vg = vgicp.align(*gmap, *source, *gmap, init);
  
  // VGICP 시간 측정 종료
  auto vgicp_end = std::chrono::high_resolution_clock::now();
  auto vgicp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vgicp_end - vgicp_start);

  std::cout << "Running GenZ-VGICP..." << std::endl;
  // GenZ-VGICP용 새로운 GaussianVoxelMap 생성
  auto gmap_genz = create_gaussian_voxelmap(*target, voxel_size);
  gmap_genz->set_search_offsets(27);  // GenZ-VGICP용 설정

  // GenZ-VGICP 시간 측정 시작
  auto genz_start = std::chrono::high_resolution_clock::now();
  
  auto res_gz = genz_vgicp.align(*gmap_genz, *source, *gmap_genz, init);
  
  // GenZ-VGICP 시간 측정 종료
  auto genz_end = std::chrono::high_resolution_clock::now();
  auto genz_duration = std::chrono::duration_cast<std::chrono::milliseconds>(genz_end - genz_start);

  // 8) 결과 출력
  std::cout << "\n--- VGICP ---\n"
            << "RMSE: " << res_vg.error << "\n"
            << "Transform:\n" << res_vg.T_target_source.matrix() << "\n"
            << "Iterations: " << res_vg.iterations << "\n"
            << "Converged: " << (res_vg.converged ? "Yes" : "No") << "\n"
            << "Execution time: " << vgicp_duration.count() << " ms\n"
            << "Time per iteration: " << vgicp_duration.count() / res_vg.iterations << " ms\n";

  std::cout << "\n--- GenZ-VGICP ---\n"
            << "RMSE: " << res_gz.error << "\n"
            << "Transform:\n" << res_gz.T_target_source.matrix() << "\n"
            << "Iterations: " << res_gz.iterations << "\n"
            << "Converged: " << (res_gz.converged ? "Yes" : "No") << "\n"
            << "Execution time: " << genz_duration.count() << " ms\n"
            << "Time per iteration: " << genz_duration.count() / res_gz.iterations << " ms\n";

  // 시간 비교 출력
  std::cout << "\n--- 시간 비교 ---\n"
            << "VGICP vs GenZ-VGICP 속도 비율: " 
            << static_cast<double>(vgicp_duration.count()) / genz_duration.count() << "x\n"
            << "(>1이면 GenZ-VGICP가 더 빠름)\n\n";

  // 9) 두 변환의 차이 계산
  Eigen::Isometry3d delta = res_gz.T_target_source * res_vg.T_target_source.inverse();
  std::cout << "\n--- 변환 차이 ---\n" << delta.matrix() << "\n";

  // 10) 정합된 source 포인트 클라우드를 변환하여 병합 결과를 PLY로 저장
  pcl::PointCloud<pcl::PointXYZ>::Ptr genz_cloud(new pcl::PointCloud<pcl::PointXYZ>());        // GenZ-VGICP 결과
  pcl::PointCloud<pcl::PointXYZ>::Ptr vgicp_cloud(new pcl::PointCloud<pcl::PointXYZ>());       // VGICP 결과
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged_genz(new pcl::PointCloud<pcl::PointXYZ>());       // GenZ-VGICP 병합
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged_vgicp(new pcl::PointCloud<pcl::PointXYZ>());      // VGICP 병합
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  // target 포인트 저장 (병합용)
  for (const auto& pt : target_points) {
    pcl::PointXYZ p;
    p.x = pt[0]; p.y = pt[1]; p.z = pt[2];
    target_cloud->push_back(p);
    merged_genz->push_back(p);    // GenZ 병합에 추가
    merged_vgicp->push_back(p);   // VGICP 병합에 추가
  }

  // source 포인트 변환 및 저장
  for (const auto& pt : source_points) {
    Eigen::Vector4d p = pt.cast<double>();
    
    // 원본 source 저장
    pcl::PointXYZ pcl_source;
    pcl_source.x = pt[0];
    pcl_source.y = pt[1];
    pcl_source.z = pt[2];
    source_cloud->push_back(pcl_source);
    
    // GenZ-VGICP 결과
    Eigen::Vector4d p_trans_genz = res_gz.T_target_source * p;
    pcl::PointXYZ pcl_p_genz;
    pcl_p_genz.x = p_trans_genz[0];
    pcl_p_genz.y = p_trans_genz[1];
    pcl_p_genz.z = p_trans_genz[2];
    genz_cloud->push_back(pcl_p_genz);     // 결과만 저장
    merged_genz->push_back(pcl_p_genz);    // 병합에 추가
    
    // VGICP 결과
    Eigen::Vector4d p_trans_vgicp = res_vg.T_target_source * p;
    pcl::PointXYZ pcl_p_vgicp;
    pcl_p_vgicp.x = p_trans_vgicp[0];
    pcl_p_vgicp.y = p_trans_vgicp[1];
    pcl_p_vgicp.z = p_trans_vgicp[2];
    vgicp_cloud->push_back(pcl_p_vgicp);    // 결과만 저장
    merged_vgicp->push_back(pcl_p_vgicp);   // 병합에 추가
  }

  // PLY 파일로 저장
  pcl::io::savePLYFile("genz_result.ply", *genz_cloud);              // GenZ-VGICP 결과만
  pcl::io::savePLYFile("vgicp_result.ply", *vgicp_cloud);           // VGICP 결과만
  pcl::io::savePLYFile("merged_genz_vgicp.ply", *merged_genz);      // GenZ-VGICP 병합
  pcl::io::savePLYFile("merged_vgicp.ply", *merged_vgicp);          // VGICP 병합
  pcl::io::savePLYFile("target.ply", *target_cloud);                // 원본 target
  pcl::io::savePLYFile("source.ply", *source_cloud);                // 원본 source
  
  std::cout << "\n결과 파일 저장 완료:" << std::endl;
  std::cout << "1. 변환 결과만:" << std::endl;
  std::cout << "  - GenZ-VGICP: genz_result.ply" << std::endl;
  std::cout << "  - VGICP: vgicp_result.ply" << std::endl;
  std::cout << "2. 병합 결과:" << std::endl;
  std::cout << "  - GenZ-VGICP: merged_genz_vgicp.ply" << std::endl;
  std::cout << "  - VGICP: merged_vgicp.ply" << std::endl;
  std::cout << "3. 원본 데이터:" << std::endl;
  std::cout << "  - Target: target.ply" << std::endl;
  std::cout << "  - Source: source.ply" << std::endl;

  return 0;
}