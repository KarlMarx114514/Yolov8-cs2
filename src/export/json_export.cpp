#include "export/json_export.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

void cs2_export::exportToJSON(const std::vector<cs2_detection::Detection>& detections, 
                             const std::string& filename) {
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        std::cerr << "Error: Cannot create JSON file " << filename << std::endl;
        return;
    }
    
    json_file << "{\n";
    json_file << "  \"detections\": [\n";
    
    for (int i = 0; i < static_cast<int>(detections.size()); i++) {
        const auto& det = detections[i];
        json_file << "    {\n";
        json_file << "      \"bbox\": {\n";
        json_file << "        \"x\": " << det.bbox.x << ",\n";
        json_file << "        \"y\": " << det.bbox.y << ",\n";
        json_file << "        \"width\": " << det.bbox.width << ",\n";
        json_file << "        \"height\": " << det.bbox.height << "\n";
        json_file << "      },\n";
        json_file << "      \"confidence\": " << det.confidence << ",\n";
        json_file << "      \"class_id\": " << det.class_id << ",\n";
        json_file << "      \"class_name\": \"" << det.class_name << "\"\n";
        json_file << "    }";
        if (i < static_cast<int>(detections.size()) - 1) json_file << ",";
        json_file << "\n";
    }
    
    json_file << "  ]\n";
    json_file << "}\n";
    
    json_file.close();
    std::cout << "CS2 detection data exported to: " << filename << std::endl;
}