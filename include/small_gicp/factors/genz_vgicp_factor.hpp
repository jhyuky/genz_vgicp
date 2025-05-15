// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

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
    double voxel_size;  // voxel_size 멤버 변수 추가
  };

  /// @brief Constructor
  GenZVGICPFactor(const Setting& = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      mahalanobis(Eigen::Matrix4d::Zero()),
      alpha_v(0.5),               // 보셀 평면성 가중치 (α_v), 기본 VGICP
      alpha_g(0.5)                // 대응 평면성 가중치 (α_G), 기본 50:50
  {}

  size_t target_index;             ///< Target point index
  size_t source_index;             ///< Source point index
  Eigen::Matrix4d mahalanobis;     ///< Fused precision matrix
  double alpha_v;                  ///< Voxel-level weight (0=GenZ, 1=VGICP)
  double alpha_g;                  ///< Correspondence-level weight (0=P2Pt, 1=P2Pl)

  bool inlier() const {
    return target_index != std::numeric_limits<size_t>::max();
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
      Eigen::Matrix<double, 6, 6>* H,
      Eigen::Matrix<double, 6, 1>* b,
      double* e) {
    // 인덱스 초기화
    this->source_index = source_idx;
    this->target_index = std::numeric_limits<size_t>::max();

    // 소스 점 변환
    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);

    // 최근접 이웃 탐색
    size_t t_idx;
    double sq_dist;
    if (!traits::nearest_neighbor_search(target_tree, transed_source_pt, &t_idx, &sq_dist)
        || rejector(target, source, T, t_idx, source_index, sq_dist)) {
      return false;
    }
    target_index = t_idx;

    // 공분산 합산 후 precision 계산 (VGICP)
    Eigen::Matrix4d RCR =
        traits::cov(target, target_index)
      + T.matrix() * traits::cov(source, source_index) * T.matrix().transpose();
    mahalanobis.block<3, 3>(0, 0) = RCR.block<3, 3>(0, 0).inverse();

    // 잔차 및 야코비안 준비
    Eigen::Vector4d residual = traits::point(target, target_index) - transed_source_pt;
    Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
    J.block<3, 3>(0, 0) = T.linear() * skew(
        traits::point(source, source_index).template head<3>());
    J.block<3, 3>(0, 3) = -T.linear();

    // 1) VGICP D2D 성분
    Eigen::Matrix<double, 6, 6> H_d2d = J.transpose() * mahalanobis * J;
    Eigen::Matrix<double, 6, 1> b_d2d = J.transpose() * mahalanobis * residual;
    double e_d2d = 0.5 * residual.transpose() * mahalanobis * residual;

    // 2) GenZ P2Pl / P2Pt 성분
    // 2a) 법선 계산 (voxel covariance 활용)
    Eigen::Matrix3d cov3 = traits::cov(target, target_index).template block<3, 3>(0, 0);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov3);
    Eigen::Vector3d normal = es.eigenvectors().col(0);
    
    // 3D residual / jacobian
    auto res3 = residual.head<3>();
    Eigen::Matrix<double, 3, 6> J3 = J.block<3, 6>(0, 0);

    // P2Pl
    double r_pl = normal.dot(res3);                     
    Eigen::RowVector<double, 6> J_pl = normal.transpose() * J3;
    Eigen::Matrix<double, 6, 6> H_pl = J_pl.transpose() * J_pl;
    Eigen::Matrix<double, 6, 1> b_pl = -J_pl.transpose() * r_pl;
    double e_pl = 0.5 * r_pl * r_pl;

    // P2Pt
    Eigen::Matrix<double, 6, 6> H_pt = J3.transpose() * J3;
    Eigen::Matrix<double, 6, 1> b_pt = -J3.transpose() * res3;
    double e_pt = 0.5 * res3.squaredNorm();

    // 3) 이중 가중 블렌딩
    // 1. alpha_v로 VGICP와 GenZ의 비율 결정
    // 2. alpha_g로 GenZ 내에서 P2Pl과 P2Pt의 비율 결정
    double w_d2d = alpha_v;                              // VGICP 가중치
    double w_genz = 1.0 - alpha_v;                       // GenZ 가중치
    double w_g_pl = w_genz * alpha_g;                    // P2Pl 가중치
    double w_g_pt = w_genz * (1.0 - alpha_g);            // P2Pt 가중치

    // 가중치 정규화
    double total_weight = w_d2d + w_g_pl + w_g_pt;
    w_d2d /= total_weight;
    w_g_pl /= total_weight;
    w_g_pt /= total_weight;

    *H = w_d2d * H_d2d + w_g_pl * H_pl + w_g_pt * H_pt;  
    *b = w_d2d * b_d2d + w_g_pl * b_pl + w_g_pt * b_pt;
    *e = w_d2d * e_d2d + w_g_pl * e_pl + w_g_pt * e_pt;

    return true;
  }

  /// @brief Evaluate error quickly (VGICP-only)
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(
      const TargetPointCloud& target,
      const SourcePointCloud& source,
      const Eigen::Isometry3d& T) const {
    if (target_index == std::numeric_limits<size_t>::max()) return 0.0;
    const Eigen::Vector4d transed_source_pt = T * traits::point(source, source_index);
    const Eigen::Vector4d res = traits::point(target, target_index) - transed_source_pt;
    return 0.5 * res.transpose() * mahalanobis * res;
  }
};

}  // namespace small_gicp
