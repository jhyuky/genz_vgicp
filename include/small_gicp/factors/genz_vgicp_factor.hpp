#pragma once

#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/points/traits.hpp>

namespace small_gicp {

/// @brief 임계값 기반 adaptive hybrid VGICP/GenZ-ICP factor
struct GenZVGICPFactor {
  struct Setting {
    Setting()
      : voxel_size(1.0),
        alpha_g(0.3),
        error_scale(1.0),
        use_combined_cov(false),
        v_min(0.3),  // 평면성 하한 임계값 (이하: GenZ-ICP만)
        v_max(0.8)   // 평면성 상한 임계값 (이상: VGICP만)
    {}

    double voxel_size;        // 복셀 크기
    double alpha_g;           // P2Pl vs P2Pt 가중치 (0=P2Pt, 1=P2Pl)
    double error_scale;       // 에러 전체 스케일 팩터
    bool use_combined_cov;    // source와 target 공분산 모두 사용 여부
    double v_min;             // 평면성 하한 임계값
    double v_max;             // 평면성 상한 임계값
  };

  /// @brief 생성자
  GenZVGICPFactor(const Setting& s = Setting())
    : target_index(std::numeric_limits<size_t>::max()),
      source_index(std::numeric_limits<size_t>::max()),
      mahalanobis(Eigen::Matrix4d::Zero()),
      alpha_g(s.alpha_g),
      error_scale(s.error_scale),
      voxel_size(s.voxel_size),
      use_combined_cov(s.use_combined_cov),
      v_min(s.v_min),
      v_max(s.v_max)
  {}

  size_t target_index;
  size_t source_index;
  Eigen::Matrix4d mahalanobis;
  double alpha_g;
  double error_scale;
  double voxel_size;
  bool use_combined_cov;
  double v_min;  // 추가: 평면성 하한 임계값
  double v_max;  // 추가: 평면성 상한 임계값

  bool inlier() const {
    return target_index != std::numeric_limits<size_t>::max();
  }

  /// @brief 평면성 기반 adaptive alpha_v 계산 함수 (임계값 기반)
  double compute_adaptive_alpha_v(const Eigen::Matrix3d& cov) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    auto eigs = es.eigenvalues();
    double planarity = eigs(2) / (eigs(0) + eigs(1) + eigs(2));
    // 임계값 기반 분기
    if (planarity < v_min) return 0.0;      // GenZ-ICP만
    if (planarity > v_max) return 1.0;      // VGICP만
    // 중간 구간: 선형 보간
    return (planarity - v_min) / (v_max - v_min);
  }

  /// @brief normal 계산 (기존과 동일)
  Eigen::Vector3d compute_normal(
      const Eigen::Matrix3d& cov,
      const Eigen::Vector3d& point_diff) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d n = es.eigenvectors().col(0);
    if (n.dot(point_diff) > 0) n = -n;
    return n;
  }

  /// @brief linearize 함수 (adaptive alpha_v 적용)
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

    // 최근접 이웃 검색
    Eigen::Vector4d p_s = T * traits::point(source, source_index);
    size_t t_idx; double sqd;
    if (!traits::nearest_neighbor_search(target_tree, p_s, &t_idx, &sqd)) {
      return false;
    }
    if (rejector(target, source, T, t_idx, source_index, sqd)) {
      return false;
    }
    target_index = t_idx;

    // VGICP D2D
    Eigen::Matrix4d C_t = traits::cov(target, target_index);
    Eigen::Matrix4d C_s = T.matrix() * traits::cov(source, source_index) * T.matrix().transpose();
    Eigen::Matrix4d RCR = C_t + C_s;
    const double lambda = 1e-6;
    mahalanobis.block<3,3>(0,0) = (RCR.block<3,3>(0,0) + lambda * Eigen::Matrix3d::Identity()).inverse();

    Eigen::Vector4d residual = traits::point(target, target_index) - p_s;
    Eigen::Matrix<double,4,6> J4 = Eigen::Matrix<double,4,6>::Zero();
    Eigen::Vector3d source_point = traits::point(source, source_index).template head<3>();
    J4.block<3,3>(0,0) = T.linear() * skew(source_point);
    J4.block<3,3>(0,3) = -T.linear();

    Eigen::Matrix<double,6,6> H_d2d = J4.transpose() * mahalanobis * J4;
    Eigen::Matrix<double,6,1> b_d2d = J4.transpose() * mahalanobis * residual;
    double e_d2d = 0.5 * residual.transpose() * mahalanobis * residual;

    // GenZ P2Pl / P2Pt
    Eigen::Vector3d res3 = residual.head<3>();
    Eigen::Matrix3d cov3;
    if (use_combined_cov) {
        cov3 = (C_t + C_s).template block<3,3>(0,0);
    } else {
        cov3 = C_t.template block<3,3>(0,0);
    }

    // correspondence마다 adaptive alpha_v 계산
    double alpha_v = compute_adaptive_alpha_v(cov3);

    Eigen::Vector3d normal = compute_normal(cov3, res3);
    Eigen::Matrix<double,3,6> J3 = J4.block<3,6>(0,0);
    double r_pl = normal.dot(res3) / voxel_size;
    Eigen::RowVector<double,6> J_pl = (normal.transpose() * J3) / voxel_size;
    Eigen::Matrix<double,6,6> H_pl = J_pl.transpose() * J_pl;
    Eigen::Matrix<double,6,1> b_pl = -J_pl.transpose() * r_pl;
    double e_pl = 0.5 * r_pl * r_pl;

    Eigen::Matrix<double,6,6> H_pt = (J3.transpose() * J3) / (voxel_size*voxel_size);
    Eigen::Matrix<double,6,1> b_pt = -J3.transpose() * res3 / (voxel_size*voxel_size);
    double e_pt = 0.5 * res3.squaredNorm() / (voxel_size*voxel_size);

    // adaptive alpha_v로 가중치 블렌딩
    double w_d2d  = alpha_v;         // VGICP의 비중
    double w_genz = 1.0 - alpha_v;   // GenZ의 비중
    double w_pl   = w_genz * alpha_g;
    double w_pt   = w_genz * (1.0 - alpha_g);

    *H = w_d2d*H_d2d + w_pl*H_pl + w_pt*H_pt;
    *b = w_d2d*b_d2d + w_pl*b_pl + w_pt*b_pt;
    *e = error_scale * (w_d2d*e_d2d + w_pl*e_pl + w_pt*e_pt);
    return true;
  }

  /// @brief error 함수 (adaptive alpha_v 적용)
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

    // correspondence마다 adaptive alpha_v 계산
    double alpha_v = compute_adaptive_alpha_v(cov3);

    double e_pl = 0.5 * std::pow(res3.dot(compute_normal(cov3, res3)) / voxel_size, 2);
    double e_pt = 0.5 * (res3.squaredNorm() / (voxel_size*voxel_size));

    double w_d2d  = alpha_v;
    double w_genz = 1.0 - alpha_v;
    double w_pl   = w_genz * alpha_g;
    double w_pt   = w_genz * (1.0 - alpha_g);

    return error_scale * (w_d2d*e_d2d + w_pl*e_pl + w_pt*e_pt);
  }
};

}  // namespace small_gicp
