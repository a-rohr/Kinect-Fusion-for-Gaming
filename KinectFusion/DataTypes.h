#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

namespace kinectfusion {

    struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;
        CameraParameters level(const size_t level) const
        {
            if (level == 0) return *this;
            const float scale_factor = powf(0.5f, static_cast<float>(level));
            return CameraParameters { image_width >> level, image_height >> level,
                                      focal_x * scale_factor, focal_y * scale_factor,
                                      (principal_x + 0.5f) * scale_factor - 0.5f,
                                      (principal_y + 0.5f) * scale_factor - 0.5f };
        }
	};
}
