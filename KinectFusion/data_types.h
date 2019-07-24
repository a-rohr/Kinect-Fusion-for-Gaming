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

    struct GlobalConfiguration {
        //int3 volume_size { make_int3(256, 256, 256) };
        float voxel_scale { 2.f };
        int bfilter_kernel_size { 5 };
        float bfilter_color_sigma { 1.f };
        float bfilter_spatial_sigma { 1.f };
        float init_depth { 1000.f };
        bool use_output_frame = { true };
        float truncation_distance { 25.f };
        float depth_cutoff_distance { 1000.f };
        int num_levels { 3 };
        int triangles_buffer_size { 3 * 2000000 };
        int pointcloud_buffer_size { 3 * 2000000 };
        float distance_threshold { 10.f };
        float angle_threshold { 20.f };
        std::vector<int> icp_iterations {10, 5, 4};
    };
}
