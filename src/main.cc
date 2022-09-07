#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>
#include <iostream>
#include <cnpy.h>

class Utils{
  public:
    template <typename T>
    static void SaveNpy(const std::string& filename, const T* data, const std::vector<size_t>& shape)
    {
        cnpy::npy_save<T>(filename, data, shape);
    }

    template <typename T>
    static void SaveNpy(const std::string& filename, const std::vector<T>& data, const std::vector<size_t>& shape)
    {
        SaveNpy<T>(filename, &data[0], shape);
    }

    template <typename T>
    static void SaveNpy(const std::string& filename, const cv::Mat& mat, const std::vector<size_t>& shape)
    {
        cv::Mat t = mat;
        if (!mat.isContinuous()) {
            t = mat.clone();
        }
        auto* ptr = t.ptr<T>(0);
        SaveNpy<T>(filename, ptr, shape);
    }
    template <typename T> static void SaveNpy(const std::string& filename, const cv::Mat& mat, int dim3 = -1)
    {
        std::vector<size_t> shape;
        shape.push_back(mat.rows);
        shape.push_back(mat.cols);
        if (dim3 > 0) {
            shape.push_back(dim3);
        }
        SaveNpy<T>(filename, mat, shape);
    }
};

cv::Mat preprocess_c(cv::Mat& img){
    auto long_edge = std::max(img.cols, img.rows);
    auto short_edge = std::min(img.cols, img.rows);

    float long_frac = 1.0 * 2048 / long_edge;
    float short_frac = 1.0 * 1024 / short_edge;
    float frac = long_frac > short_frac ? short_frac : long_frac;

    auto h_resize = round(frac * img.rows);
    auto w_resize = round(frac * img.cols);

    auto h_pad = ceil(h_resize / 32) * 32;
    auto w_pad = ceil(w_resize / 32) * 32;

    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(w_resize, h_resize));
    img_resize.convertTo(img_resize, CV_32F);
    cv::cvtColor(img_resize, img_resize, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_channel(3);
    cv::split(img_resize, rgb_channel);
    rgb_channel[0] = (rgb_channel[0] - 123.675) / 58.395;
    rgb_channel[1] = (rgb_channel[1] - 116.28) / 57.12;
    rgb_channel[2] = (rgb_channel[2] - 103.53) / 57.375;
    cv::merge(rgb_channel, img_resize);

    cv::Mat res;
    cv::copyMakeBorder(img_resize, res, 0, h_pad - h_resize, 0, w_pad - w_resize,
                       cv::BORDER_CONSTANT);
    return res;
}

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    cv::Mat imgs = cv::imread("../imgs/282555,d1d580004930ad0f.jpg");
    cv::cvtColor(imgs, imgs, cv::COLOR_BGR2RGB);

    cv::Mat preprocess_res = preprocess_c(imgs);
    
    std::string filename = "WriteMat_imgs.npy";
    Utils::SaveNpy<float>(filename, preprocess_res, 3);
    return EXIT_SUCCESS;
}
