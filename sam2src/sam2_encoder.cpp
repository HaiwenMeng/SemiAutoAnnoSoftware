/**
  @Author: mpj
  @Date  : 2026/4/6 17:13
  @version V1.0
  @since C++17
**/
#include "sam2_encoder.h"

Sam2::Encoder::Encoder(const std::string& model_path, int cuda_device_id) {
    // stream
    if (!m_stream) {
        checkRuntime(cudaStreamCreate(&m_stream));
    }
    // device
    checkRuntime(cudaSetDevice(cuda_device_id));
    if (initialize_trt(model_path) == false) {
        ERROR("Failed to initialize TensorRT engine for SAM Encoder");
    }
}

Sam2::Encoder::~Encoder() {
    if (m_stream) {
        checkRuntime(cudaStreamDestroy(m_stream));
    }
}

bool Sam2::Encoder::initialize_trt(const std::string& model_path) {
    if (model_path.empty()) {
        ERROR("engine_path is empty!");
        return false;
    }
    m_trt = trt::load(model_path);
    if (m_trt == nullptr) {
        ERROR("load engine failed!");
        return false;
    }

    m_trt->print();

    input_names = m_trt->get_input_names();
    output_names = m_trt->get_output_names();
    if (input_names.size() != 1 || output_names.size() != 3) {
        ERROR("The number of input should be 1 and output should be 3, but got %d and %d", input_names.size(), output_names.size());
        return false;
    }

    m_input_dims = m_trt->static_dims(input_names[0].c_str());
    m_image_embed_dims = m_trt->static_dims(output_names[0].c_str());
    m_high_res_feats0_dims = m_trt->static_dims(output_names[1].c_str());
    m_high_res_feats1_dims = m_trt->static_dims(output_names[2].c_str());
    if (m_input_dims.size() != 4 || m_image_embed_dims.size() != 4) {
        ERROR(" The input dims should be 4 and output dims should be 4");
        ERROR("Input dims: %s, Output dims: %s", trt::format_shape(m_input_dims).c_str(),
              trt::format_shape(m_image_embed_dims).c_str());
        return false;
    }

    // 分配输入输出缓冲区
    size_t input_numel = vectorProduct(m_input_dims);
    size_t image_embed_numel = vectorProduct(m_image_embed_dims);
    size_t high_res_feats0_numel = vectorProduct(m_high_res_feats0_dims);
    size_t high_res_feats1_numel = vectorProduct(m_high_res_feats1_dims);
    m_input_buffer.gpu(input_numel);
    m_image_embed_buffer.gpu(image_embed_numel);
    m_high_res_feats0_buffer.gpu(high_res_feats0_numel);
    m_high_res_feats1_buffer.gpu(high_res_feats1_numel);

    return true;
}

Sam2::ImageEmbedding Sam2::Encoder::forward(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        ERROR("Input image is empty");
        throw std::runtime_error("Input image is empty");
    }
    if (is_image_embedding_ready()) {
        return {
            m_image_embed_buffer.gpu(),
            m_high_res_feats0_buffer.gpu(),
            m_high_res_feats1_buffer.gpu(),
        };
    }

    // preprocess image
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    cv::Mat resized = ImagePreprocessor::resize_fp32(rgb_image, target_size_);
    cv::Mat input = ImagePreprocessor::hwc2chw_fp32(resized);

    // copy tensor to gpu
    size_t input_numel = vectorProduct(m_input_dims);
    size_t image_embed_numel = vectorProduct(m_image_embed_dims);
    size_t high_res_feats0_numel = vectorProduct(m_high_res_feats0_dims);
    size_t high_res_feats1_numel = vectorProduct(m_high_res_feats1_dims);
    float* input_device = m_input_buffer.gpu();
    checkRuntime(
        cudaMemcpyAsync(input_device, input.data, input_numel * sizeof(float),
            cudaMemcpyHostToDevice, m_stream ));

    // infer
    float* image_embed_device = m_image_embed_buffer.gpu();
    float* high_res_feats0_device = m_high_res_feats0_buffer.gpu();
    float* high_res_feats1_device = m_high_res_feats1_buffer.gpu();
    std::map<std::string, void*> bindings = {
        {input_names[0].c_str(), input_device},
        {output_names[0].c_str(), image_embed_device},
        {output_names[1].c_str(), high_res_feats0_device},
        {output_names[2].c_str(), high_res_feats1_device},
    };
    if (!m_trt->forward(bindings, m_stream)) {
        ERROR("Failed to tensorRT forward");
        throw std::runtime_error("TensorRT forward failed");
    }

    // sync stream，不需要复制回cpu
    checkRuntime(cudaStreamSynchronize(m_stream));

    is_image_embedding_ready_ = true;
    return {
        m_image_embed_buffer.gpu(),
        m_high_res_feats0_buffer.gpu(),
        m_high_res_feats1_buffer.gpu(),
    };
}

Sam2::ImageEmbedding Sam2::Encoder::get_image_embedding() const {
    if (!is_image_embedding_ready()) {
        ERROR("Image embedding is not ready");
        throw std::runtime_error("Image embedding is not ready");
    }
    return {
        m_image_embed_buffer.gpu(),
        m_high_res_feats0_buffer.gpu(),
        m_high_res_feats1_buffer.gpu(),
    };
}

bool Sam2::Encoder::reset_image_embedding() {
    is_image_embedding_ready_ = false;
    cudaMemset(m_image_embed_buffer.gpu(), 0, vectorProduct(m_image_embed_dims) * sizeof(float));
    cudaMemset(m_high_res_feats0_buffer.gpu(), 0, vectorProduct(m_high_res_feats0_dims) * sizeof(float));
    cudaMemset(m_high_res_feats1_buffer.gpu(), 0, vectorProduct(m_high_res_feats1_dims) * sizeof(float));
    return true;
}
