// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>             // ★ 추가: 플래너리티 계산을 위한 SelfAdjointEigenSolver

#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>

namespace small_gicp {

/// @brief Gaussian voxel that computes and stores voxel mean, covariance, and planarity weight.
struct GaussianVoxel {
public:
  struct Setting {};

  /// @brief Constructor.
  GaussianVoxel()
    : finalized(false),
      num_points(0),
      mean(Eigen::Vector4d::Zero()),
      cov(Eigen::Matrix4d::Zero()),
      alpha(1.0)                // ★ 추가: adaptive weight 초기값 1.0 (VGICP 모드)
  {}

  /// @brief Number of points in the voxel.
  size_t size() const { return 1; }

  /// @brief Add a point to the voxel.
  /// 기존 평균 및 공분산 재계산을 위해 임시로 sum 상태로 되돌림
  template <typename PointCloud>
  void add(const Setting& setting,
           const Eigen::Vector4d& transformed_pt,
           const PointCloud& points,
           size_t i,
           const Eigen::Isometry3d& T) {
    if (finalized) {
      // 이전 finalize()가 호출되었으므로 sum 상태로 복원
      this->finalized = false;
      this->mean *= num_points;
      this->cov  *= num_points;
    }

    num_points++;
    this->mean += transformed_pt;  // 포인트 평균 합산
    this->cov  += T.matrix() * traits::cov(points, i) * T.matrix().transpose();  // 공분산 합산
  }

  /// @brief Finalize the voxel: compute mean, covariance, and planarity weight.
  void finalize() {
    if (finalized) {
      return;
    }

    // 평균 및 공분산 산술평균
    mean /= num_points;
    cov  /= num_points;

    // ★ planarity 계산
    // 3×3 좌표 블록의 고유값 분해로 평면성 지표 p = (λ1 - λ0)/λ2
    Eigen::Matrix3d cov3 = cov.block<3, 3>(0, 0);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov3);
    auto evals = es.eigenvalues();  // 오름차순: evals[0] ≤ evals[1] ≤ evals[2]
    double planarity = (evals[1] - evals[0]) / evals[2];
    // clamp로 0~1 범위로 제한하여 alpha 설정
    alpha = std::clamp(planarity, 0.0, 1.0);

    finalized = true;
  }

public:
  bool finalized;           ///< If true, mean and cov are finalized
  size_t num_points;        ///< Number of input points
  Eigen::Vector4d mean;     ///< Sum or mean of transformed points
  Eigen::Matrix4d cov;      ///< Sum or mean of covariances
  double alpha;             ///< Adaptive planarity weight for GenZVGICP
};

namespace traits {

template <>
struct Traits<GaussianVoxel> {
  static size_t size(const GaussianVoxel& voxel) { return 1; }
  static bool has_points(const GaussianVoxel& voxel) { return true; }
  static bool has_covs(const GaussianVoxel& voxel)   { return true; }

  static const Eigen::Vector4d& point(const GaussianVoxel& voxel, size_t) {
    return voxel.mean;
  }
  static const Eigen::Matrix4d& cov(const GaussianVoxel& voxel, size_t) {
    return voxel.cov;
  }

  static size_t nearest_neighbor_search(const GaussianVoxel& voxel,
                                        const Eigen::Vector4d& pt,
                                        size_t* k_index,
                                        double* k_sq_dist) {
    *k_index = 0;
    *k_sq_dist = (voxel.mean - pt).squaredNorm();
    return 1;
  }

  static size_t knn_search(const GaussianVoxel& voxel,
                           const Eigen::Vector4d& pt,
                           size_t,
                           size_t* k_index,
                           double* k_sq_dist) {
    return nearest_neighbor_search(voxel, pt, k_index, k_sq_dist);
  }

  template <typename Result>
  static void knn_search(const GaussianVoxel& voxel,
                         const Eigen::Vector4d& pt,
                         Result& result) {
    result.push(0, (voxel.mean - pt).squaredNorm());
  }
};

}  // namespace traits

using GaussianVoxelMap = IncrementalVoxelMap<GaussianVoxel>;

}  // namespace small_gicp