#pragma once
#include "core/detection_types.hpp"
#include <vector>
#include <string>

namespace cs2_export{
    void exportToJSON(const std::vector<cs2_detection::Detection>& detections, const std::string& filename);
}