#include "lio/scan_dataset_exporter.h"

#include "basic/logs.h"
#include "lio/params.h"

#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace BASIC;

namespace LI2Sup {
namespace {

class ScanDatasetWriter {
public:
  void write(const CloudPtr& cloud_in_imu, const NavState& state) {
    if (!g_save_map || !cloud_in_imu || cloud_in_imu->empty()) return;
    if (!init()) return;

    const double stamp = state.timestamp;
    const long sec = static_cast<long>(std::floor(stamp));
    long nsec = static_cast<long>(std::llround((stamp - static_cast<double>(sec)) * 1e9));
    const std::string file = makeName(index_, sec, nsec);

    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(cloud_in_imu->size());

    const M3 R_IL = g_lidar_imu.R_;
    const V3 t_IL = g_lidar_imu.t_;
    const M3 R_LI = R_IL.transpose();

    for (const auto& pt : cloud_in_imu->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      const V3 p_lidar = R_LI * (V3(pt.x, pt.y, pt.z) - t_IL);
      pcl::PointXYZI out;
      out.x = static_cast<float>(p_lidar.x());
      out.y = static_cast<float>(p_lidar.y());
      out.z = static_cast<float>(p_lidar.z());
      out.intensity = pt.intensity;
      cloud.push_back(out);
    }

    if (cloud.empty()) return;
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;

    if (pcl::io::savePCDFileBinary(clouds_dir_ + "/" + file, cloud) != 0) {
      LOG(WARNING) << RED << " ---> [Scan dataset] Failed to save: " << file << RESET;
      return;
    }

    const M3 R_WL = state.R.R_ * R_IL;
    const V3 t_WL = state.R.R_ * t_IL + state.p;
    Eigen::Quaterniond q(R_WL.cast<double>());
    q.normalize();

    poses_csv_ << index_ << "," << sec << "," << nsec << ","
               << std::setprecision(12)
               << t_WL.x() << "," << t_WL.y() << "," << t_WL.z() << ","
               << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ","
               << file << "\n";

    poses_tum_ << std::fixed << std::setprecision(9) << stamp << " "
               << std::setprecision(12)
               << t_WL.x() << " " << t_WL.y() << " " << t_WL.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

    poses_csv_.flush();
    poses_tum_.flush();
    index_++;
  }

private:
  bool init() {
    if (initialized_) return true;
    out_dir_ = g_save_map_dir + "/scan_dataset";
    clouds_dir_ = out_dir_ + "/clouds";

    std::error_code ec;
    std::filesystem::remove_all(out_dir_, ec);
    ec.clear();
    std::filesystem::create_directories(clouds_dir_, ec);
    if (ec) return false;

    poses_csv_.open(out_dir_ + "/poses_lidar.csv", std::ios::out);
    poses_tum_.open(out_dir_ + "/poses_lidar_tum.txt", std::ios::out);
    if (!poses_csv_.is_open() || !poses_tum_.is_open()) return false;

    poses_csv_ << "frame_index,stamp_sec,stamp_nsec,tx,ty,tz,qx,qy,qz,qw,cloud_file\n";
    initialized_ = true;
    LOG(INFO) << GREEN << " ---> [Scan dataset] Writing scans to " << out_dir_ << RESET;
    return true;
  }

  std::string makeName(std::size_t index, long sec, long nsec) const {
    std::ostringstream ss;
    ss << "frame_" << std::setw(6) << std::setfill('0') << index
       << "_" << std::setw(10) << std::setfill('0') << sec
       << "_" << std::setw(9) << std::setfill('0') << nsec
       << ".pcd";
    return ss.str();
  }

  bool initialized_ = false;
  std::size_t index_ = 0;
  std::string out_dir_;
  std::string clouds_dir_;
  std::ofstream poses_csv_;
  std::ofstream poses_tum_;
};

ScanDatasetWriter& writer() {
  static ScanDatasetWriter w;
  return w;
}

}  // namespace

void ExportScanDataset(const CloudPtr& imu_frame_cloud, const NavState& state) {
  writer().write(imu_frame_cloud, state);
}

}  // namespace LI2Sup
