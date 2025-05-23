// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>         // 추가: 평면 법선 추출용 SelfAdjointEigenSolver
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief Adaptive-Weighted VGICP (GenZ) per-point error factor.
struct GenZVGICPFactor {
  struct Setting {
    Setting()
      : voxel_size(1.0),
        alpha_v(0.2),
        alpha_g(0.8),
        error_scale(1.0),
        use_combined_cov(false)
    {}

    double voxel_size;        // 복셀 크기
    double alpha_v;           // VGICP vs GenZ 가중치 (0=GenZ, 1=VGICP)
    double alpha_g;           // P2Pl vs P2Pt 가중치 (0=P2Pt, 1=P2Pl)
    double error_scale;       // 에러 전체 스케일 팩터
    bool use_combined_cov;    // source와 target 공분산 모두 사용 여부
  };

  /// @brief Constructor
  GenZVGICPFactor(const Setting& s = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      mahalanobis(Eigen::Matrix4d::Zero()),
      alpha_v(s.alpha_v),
      alpha_g(s.alpha_g),
      error_scale(s.error_scale),
      voxel_size(s.voxel_size),
      use_combined_cov(s.use_combined_cov)
  {}

  size_t target_index;             ///< Target point index
  size_t source_index;             ///< Source point index
  Eigen::Matrix4d mahalanobis;     ///< Fused precision matrix
  double alpha_v;                  ///< Voxel-level weight
  double alpha_g;                  ///< Correspondence-level weight
  double error_scale;             ///< Error scale normalization factor
  double voxel_size;              ///< Voxel size for neighborhood search
  bool use_combined_cov;          ///< Use combined covariance

  bool inlier() const {
    return target_index != std::numeric_limits<size_t>::max();
  }

  /// @brief Compute normal with consistent orientation
  Eigen::Vector3d compute_normal(
      const Eigen::Matrix3d& cov,
      const Eigen::Vector3d& point_diff) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d n = es.eigenvectors().col(0);
    // 법선 방향을 point_diff와 반대 방향으로 정렬
    if (n.dot(point_diff) > 0) n = -n;
    return n;
  }

  /// @brief Linearize the factor
  template <typename TargetPointCloud,
            typename SourcePointCloud,
            typename TargetTree,
            typename CorrespondenceRejector>
  bool linearize(
      const TargetPointCloud& target,
      const SourcePointCloud& source,
      const TargetTree& target_tree,
      const Eigen::Isometry3d& T,
      size_t source_idx,
      const CorrespondenceRejector& rejector,
      Eigen::Matrix<double,6,6>* H,
      Eigen::Matrix<double,6,1>* b,
      double* e) {
    source_index = source_idx;
    target_index = std::numeric_limits<size_t>::max();

    // 1) 최근접 이웃 검색
    Eigen::Vector4d p_s = T * traits::point(source, source_index);
    size_t t_idx; double sqd;
    
    static size_t total_attempts = 0;
    static size_t total_rejections = 0;
    
    if (!traits::nearest_neighbor_search(target_tree, p_s, &t_idx, &sqd)) {
      total_rejections++;
      if (total_attempts % 1000 == 0) {
        std::cout << "디버그: 최근접 이웃 검색 실패율: " 
                  << (double)total_rejections/total_attempts * 100.0 << "%" << std::endl;
      }
      return false;
    }
    
    total_attempts++;
    
    // rejector 조건 완화
    if (rejector(target, source, T, t_idx, source_index, sqd)) {
      total_rejections++;
      if (total_attempts % 1000 == 0) {
        std::cout << "디버그: rejector 거부율: " 
                  << (double)total_rejections/total_attempts * 100.0 
                  << "% (sqd: " << sqd << ")" << std::endl;
      }
      return false;
    }
    
    target_index = t_idx;

    // 2) VGICP D2D
    Eigen::Matrix4d C_t = traits::cov(target, target_index);
    Eigen::Matrix4d C_s = T.matrix() * traits::cov(source, source_index) * T.matrix().transpose();
    Eigen::Matrix4d RCR = C_t + C_s;
    const double lambda = 1e-6;  // Tikhonov regularization
    mahalanobis.block<3,3>(0,0) = (RCR.block<3,3>(0,0) + lambda * Eigen::Matrix3d::Identity()).inverse();

    Eigen::Vector4d residual = traits::point(target, target_index) - p_s;
    Eigen::Matrix<double,4,6> J4 = Eigen::Matrix<double,4,6>::Zero();
    
    // 수정된 부분: head() 함수 호출 방식 변경
    Eigen::Vector3d source_point = traits::point(source, source_index).template head<3>();
    J4.block<3,3>(0,0) = T.linear() * skew(source_point);
    J4.block<3,3>(0,3) = -T.linear();

    Eigen::Matrix<double,6,6> H_d2d = J4.transpose() * mahalanobis * J4;
    Eigen::Matrix<double,6,1> b_d2d = J4.transpose() * mahalanobis * residual;
    double e_d2d = 0.5 * residual.transpose() * mahalanobis * residual;

    // 3) GenZ P2Pl / P2Pt
    Eigen::Vector3d res3 = residual.head<3>();
    Eigen::Matrix3d cov3;
    if (use_combined_cov) {
        cov3 = (C_t + C_s).template block<3,3>(0,0);
    } else {
        cov3 = C_t.template block<3,3>(0,0);
    }
    Eigen::Vector3d normal = compute_normal(cov3, res3);

    Eigen::Matrix<double,3,6> J3 = J4.block<3,6>(0,0);
    // P2Pl with scale normalization
    double r_pl = normal.dot(res3) / voxel_size;
    Eigen::RowVector<double,6> J_pl = (normal.transpose() * J3) / voxel_size;
    Eigen::Matrix<double,6,6> H_pl = J_pl.transpose() * J_pl;
    Eigen::Matrix<double,6,1> b_pl = -J_pl.transpose() * r_pl;
    double e_pl = 0.5 * r_pl * r_pl;

    // P2Pt with scale normalization
    Eigen::Matrix<double,6,6> H_pt = (J3.transpose() * J3) / (voxel_size*voxel_size);
    Eigen::Matrix<double,6,1> b_pt = -J3.transpose() * res3 / (voxel_size*voxel_size);
    double e_pt = 0.5 * res3.squaredNorm() / (voxel_size*voxel_size);

    // 4) 가중치 블렌딩 (합이 1이므로 정규화 불필요)
    double w_d2d  = alpha_v;         // VGICP의 비중 (평면성이 높을수록 작아짐)
    double w_genz = 1.0 - alpha_v;   // GenZ의 비중 (평면성이 높을수록 커짐)
    double w_pl   = w_genz * alpha_g;
    double w_pt   = w_genz * (1.0 - alpha_g);

    // 디버깅 출력 추가
    std::cout << "\n--- GenZ-VGICP 디버깅 정보 ---" << std::endl;
    std::cout << "Error components:" << std::endl;
    std::cout << "e_d2d: " << e_d2d << std::endl;
    std::cout << "e_pl: " << e_pl << std::endl;
    std::cout << "e_pt: " << e_pt << std::endl;
    std::cout << "Weights:" << std::endl;
    std::cout << "w_d2d: " << w_d2d << std::endl;
    std::cout << "w_pl: " << w_pl << std::endl;
    std::cout << "w_pt: " << w_pt << std::endl;
    std::cout << "Final error: " << *e << std::endl;
    std::cout << "use_combined_cov: " << (use_combined_cov ? "true" : "false") << std::endl;

    *H = w_d2d*H_d2d + w_pl*H_pl + w_pt*H_pt;
    *b = w_d2d*b_d2d + w_pl*b_pl + w_pt*b_pt;
    *e = error_scale * (w_d2d*e_d2d + w_pl*e_pl + w_pt*e_pt);
    return true;
  }

  /// @brief Evaluate full error
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(
      const TargetPointCloud& target,
      const SourcePointCloud& source,
      const Eigen::Isometry3d& T) const {
    if (!inlier()) return 0.0;

    Eigen::Vector4d p_s = T * traits::point(source, source_index);
    Eigen::Vector4d res = traits::point(target, target_index) - p_s;
    double e_d2d = 0.5 * res.transpose() * mahalanobis * res;

    Eigen::Vector3d res3 = res.head<3>();
    Eigen::Matrix4d C_s = T.matrix() * traits::cov(source, source_index) * T.matrix().transpose();
    Eigen::Matrix3d cov3;
    if (use_combined_cov) {
        cov3 = (traits::cov(target, target_index) + C_s).template block<3,3>(0,0);
    } else {
        cov3 = traits::cov(target, target_index).template block<3,3>(0,0);
    }
    
    // P2Pl error
    double e_pl = 0.5 * std::pow(res3.dot(compute_normal(cov3, res3)) / voxel_size, 2);
    // P2Pt error (J3 기반으로 계산)
    double e_pt = 0.5 * (res3.squaredNorm() / (voxel_size*voxel_size));

    double w_d2d  = alpha_v;         // VGICP의 비중 (평면성이 높을수록 작아짐)
    double w_genz = 1.0 - alpha_v;   // GenZ의 비중 (평면성이 높을수록 커짐)
    double w_pl   = w_genz * alpha_g;
    double w_pt   = w_genz * (1.0 - alpha_g);

    return error_scale * (w_d2d*e_d2d + w_pl*e_pl + w_pt*e_pt);
  }
};

}  // namespace small_gicp