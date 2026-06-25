#pragma once

#include "common/ds.h"

namespace LI2Sup {
void ExportScanDataset(const BASIC::CloudPtr& imu_frame_cloud, const NavState& state);
}
