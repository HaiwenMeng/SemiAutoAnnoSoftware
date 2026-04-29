/**
  @Author: mpj
  @Date  : 2026/4/6 17:35
  @version V1.0
  @since C++17
**/
#include "sam2_decoder.h"

Sam2::Decoder::Decoder(const std::string& model_path, int cuda_device_id) {
    // stream
    if (!m_stream) {
        checkRuntime(cudaStreamCreate(&m_stream));
    }
    // device
    checkRuntime(cudaSetDevice(cuda_device_id));
    if (initialize_trt(model_path) == false) {
        ERROR("Failed to initialize TensorRT engine for SAM Decoder");
    }
}

Sam2::Decoder::~Decoder() {
    if (m_stream) {
        checkRuntime(cudaStreamDestroy(m_stream));
    }
}

bool Sam2::Decoder::initialize_trt(const std::string& model_path) {
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
    if (input_names.size() != 7 || output_names.size() != 2) {
        ERROR("The number of input should be 5 and output should be 2, but got %d and %d", input_names.size(),
              output_names.size());
        return false;
    }

    m_image_embed_dims = m_trt->static_dims(input_names[0].c_str());
    m_high_res_feats0_dims = m_trt->static_dims(input_names[1].c_str());
    m_high_res_feats1_dims = m_trt->static_dims(input_names[2].c_str());
    m_point_coords_dims = m_trt->static_dims(input_names[3].c_str());
    m_point_labels_dims = m_trt->static_dims(input_names[4].c_str());
    m_mask_input_dims = m_trt->static_dims(input_names[5].c_str());
    m_has_mask_input_dims = m_trt->static_dims(input_names[6].c_str());
    m_low_res_masks_dims = m_trt->static_dims(output_names[0].c_str());
    m_iou_predictions_dims = m_trt->static_dims(output_names[1].c_str());
    if (m_image_embed_dims.size() != 4 || m_high_res_feats0_dims.size() != 4 || m_high_res_feats1_dims.size() != 4 ||
        m_point_coords_dims.size() != 3 || m_point_labels_dims.size() != 2 ||
        m_mask_input_dims.size() != 4 || m_has_mask_input_dims.size() != 1 || m_iou_predictions_dims.size() != 2 ||
        m_low_res_masks_dims.size() != 4) {
        ERROR(" The input dims should be 4, 4, 4, 3, 2, 4, 1, 2, 4, 4");
        ERROR("Input dims: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
              trt::format_shape(m_image_embed_dims).c_str(),
              trt::format_shape(m_high_res_feats0_dims).c_str(),
              trt::format_shape(m_high_res_feats1_dims).c_str(),
              trt::format_shape(m_point_coords_dims).c_str(),
              trt::format_shape(m_point_labels_dims).c_str(),
              trt::format_shape(m_mask_input_dims).c_str(),
              trt::format_shape(m_has_mask_input_dims).c_str(),
              trt::format_shape(m_iou_predictions_dims).c_str(),
              trt::format_shape(m_low_res_masks_dims).c_str());
        return false;
    }

    return true;
}

Sam2::Results Sam2::Decoder::forward(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                                     const PromptPoints& prompt_points, const PromptBboxes& prompt_bboxes) {
    // memory
    int batch_size = 1;
    int num_points = prompt_points.size();
    int num_boxes = prompt_bboxes.size();
    num_points = num_points + num_boxes * 2 + 1;
    adjust_memory(batch_size, num_points);

    // image embedding tensor
    float* image_embedding_tensor = image_embedding.image_embedding_;
    float* high_res_feats0_tensor = image_embedding.high_res_feats0_;
    float* high_res_feats1_tensor = image_embedding.high_res_feats1_;

    // point coords and labels tensors
    int index = 0;
    for (const auto& [x1, y1, x2, y2]: prompt_bboxes) {
        auto transformed_top_left = ImagePreprocessor::apply_coords(
            {x1, y1}, original_image_size, target_size_);
        auto transformed_bottom_right = ImagePreprocessor::apply_coords(
            {x2, y2}, original_image_size, target_size_);
        m_point_coords_buffer.cpu()[index * 2] = transformed_top_left.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed_top_left.y;
        m_point_labels_buffer.cpu()[index] = 2.0f; // 2表示box左上角
        index++;
        m_point_coords_buffer.cpu()[index * 2] = transformed_bottom_right.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed_bottom_right.y;
        m_point_labels_buffer.cpu()[index] = 3.0f; // 3表示box右下角
        index++;
    }
    for (const auto& point: prompt_points) {
        auto transformed = ImagePreprocessor::apply_coords({point.x, point.y}, original_image_size, target_size_);
        m_point_coords_buffer.cpu()[index * 2] = transformed.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed.y;
        m_point_labels_buffer.cpu()[index] = point.label == 1 ? 1.0f : 0.0f;
        index++;
    }
    // 在torch sam2官方中，SAM会默认添加一个补齐点，坐标为(0,0)，标签为-1，表示这个点不参与计算。这里我们也做同样的处理。
    m_point_coords_buffer.cpu()[index * 2] = 0.0f;
    m_point_coords_buffer.cpu()[index * 2 + 1] = 0.0f;
    m_point_labels_buffer.cpu()[index] = -1.0f;
    m_point_coords_dims = {batch_size, static_cast<int>(prompt_points.size() + prompt_bboxes.size() * 2 + 1), 2};
    m_point_labels_dims = {batch_size, static_cast<int>(prompt_points.size() + prompt_bboxes.size() * 2 + 1)};
    checkRuntime(
        cudaMemcpyAsync(m_point_coords_buffer.gpu(), m_point_coords_buffer.cpu(), vectorProduct(m_point_coords_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_point_labels_buffer.gpu(), m_point_labels_buffer.cpu(), vectorProduct(m_point_labels_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_trt->set_run_dims(input_names[3].c_str(), m_point_coords_dims);
    m_trt->set_run_dims(input_names[4].c_str(), m_point_labels_dims);

    // Prepare mask input (zeros)
    m_mask_input_dims = {batch_size, m_mask_input_dims[1], m_mask_input_dims[2], m_mask_input_dims[3]};
    checkRuntime(
        cudaMemsetAsync(m_mask_input_buffer.gpu(), 0, vectorProduct(m_mask_input_dims) * sizeof(float), m_stream));
    m_trt->set_run_dims(input_names[5].c_str(), m_mask_input_dims);

    // Prepare has_mask_input (0 for no mask input)
    checkRuntime(
        cudaMemsetAsync(m_has_mask_input_buffer.gpu(), 0, vectorProduct(m_has_mask_input_dims) * sizeof(float), m_stream
        ));

    // infer
    const std::map<std::string, void*> bindings = {
        {input_names[0].c_str(), image_embedding_tensor},
        {input_names[1].c_str(), high_res_feats0_tensor},
        {input_names[2].c_str(), high_res_feats1_tensor},
        {input_names[3].c_str(), m_point_coords_buffer.gpu()},
        {input_names[4].c_str(), m_point_labels_buffer.gpu()},
        {input_names[5].c_str(), m_mask_input_buffer.gpu()},
        {input_names[6].c_str(), m_has_mask_input_buffer.gpu()},
        {output_names[0].c_str(), m_low_res_masks_buffer.gpu()},
        {output_names[1].c_str(), m_iou_predictions_buffer.gpu()},
    };
    if (!m_trt->forward(bindings, m_stream)) {
        ERROR("Failed to tensorRT forward");
        throw std::runtime_error("TensorRT forward failed");
    }

    // copy back
    m_iou_predictions_dims = {batch_size, m_iou_predictions_dims[1]};
    m_low_res_masks_dims = {batch_size, m_low_res_masks_dims[1], m_low_res_masks_dims[2], m_low_res_masks_dims[3]};
    checkRuntime(
        cudaMemcpyAsync(m_iou_predictions_buffer.cpu(), m_iou_predictions_buffer.gpu(), vectorProduct(
            m_iou_predictions_dims) * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_low_res_masks_buffer.cpu(), m_low_res_masks_buffer.gpu(), vectorProduct(m_low_res_masks_dims)
            * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(cudaStreamSynchronize(m_stream));

    // postprocess
    int low_res_num_masks = m_low_res_masks_dims[1];
    int low_res_height = m_low_res_masks_dims[2];
    int low_res_width = m_low_res_masks_dims[3];
    Results results;
    for (int b = 0; b < batch_size; ++b) {
        Result result;
        for (int n = 0; n < low_res_num_masks; ++n) {
            float* src_ptr = m_low_res_masks_buffer.cpu() + b * low_res_num_masks * low_res_height * low_res_width + n *
                             low_res_height * low_res_width;
            cv::Mat mask(low_res_height, low_res_width, CV_32FC1, src_ptr);
            // 放大后的mask
            auto mask_f32 = mask_postprocessing(mask, original_image_size);
            cv::Mat mask_u8;
            cv::Mat threshold_mask = mask_f32 > mask_threshold_; // CV_8UC1, 0 或 1
            threshold_mask.convertTo(mask_u8, CV_8UC1, 255.0); // 转为 0 或 1
            result.masks.push_back(mask_u8);
            // 低分辨率mask
            cv::Mat low_res_mask(low_res_height, low_res_width, CV_32FC1);
            std::memcpy(low_res_mask.data, src_ptr, low_res_height * low_res_width * sizeof(float));
            result.low_res_masks.push_back(low_res_mask);
            // IoU预测
            result.scores.push_back(m_iou_predictions_buffer.cpu()[b * low_res_num_masks + n]);
        }
        // sort masks by scores
        std::vector<int> indices(low_res_num_masks);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) {
            return result.scores[i1] > result.scores[i2];
        });
        // reorder masks and scores
        std::vector<cv::Mat> sorted_masks;
        std::vector<cv::Mat> sorted_low_res_masks;
        std::vector<float> sorted_scores;
        for (int idx: indices) {
            sorted_masks.push_back(result.masks[idx]);
            sorted_low_res_masks.push_back(result.low_res_masks[idx]);
            sorted_scores.push_back(result.scores[idx]);
        }
        result.masks = std::move(sorted_masks);
        result.low_res_masks = std::move(sorted_low_res_masks);
        result.scores = std::move(sorted_scores);

        results.emplace_back(result);
    }

    return results;
}

Sam2::Results Sam2::Decoder::forward(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                                     const PromptBboxes& prompt_bboxes) {
    // memory
    int batch_size = prompt_bboxes.size();
    int num_boxes = prompt_bboxes.size();
    adjust_memory(batch_size, 3);

    // image embedding tensor
    float* image_embedding_tensor = image_embedding.image_embedding_;
    float* high_res_feats0_tensor = image_embedding.high_res_feats0_;
    float* high_res_feats1_tensor = image_embedding.high_res_feats1_;

    // point coords and labels tensors
    int index = 0;
    for (const auto& [x1, y1, x2, y2]: prompt_bboxes) {
        auto transformed_top_left = ImagePreprocessor::apply_coords(
            {x1, y1}, original_image_size, target_size_);
        auto transformed_bottom_right = ImagePreprocessor::apply_coords(
            {x2, y2}, original_image_size, target_size_);
        m_point_coords_buffer.cpu()[index * 2] = transformed_top_left.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed_top_left.y;
        m_point_labels_buffer.cpu()[index] = 2.0f; // 2表示box左上角
        index++;
        m_point_coords_buffer.cpu()[index * 2] = transformed_bottom_right.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed_bottom_right.y;
        m_point_labels_buffer.cpu()[index] = 3.0f; // 3表示box右下角
        index++;
        // 在torch sam2官方中，SAM会默认添加一个补齐点，坐标为(0,0)，标签为-1，表示这个点不参与计算。这里我们也做同样的处理。
        // 以多box为batch维度，每一个box上根据torch sam2官方代码补充一个-1
        m_point_coords_buffer.cpu()[index * 2] = 0.0f;
        m_point_coords_buffer.cpu()[index * 2 + 1] = 0.0f;
        m_point_labels_buffer.cpu()[index] = -1.0f;
        index++;
    }
    m_point_coords_dims = {batch_size, 3, m_point_coords_dims[2]};
    m_point_labels_dims = {batch_size, 3};
    checkRuntime(
        cudaMemcpyAsync(m_point_coords_buffer.gpu(), m_point_coords_buffer.cpu(), vectorProduct(m_point_coords_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_point_labels_buffer.gpu(), m_point_labels_buffer.cpu(), vectorProduct(m_point_labels_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_trt->set_run_dims(input_names[3].c_str(), m_point_coords_dims);
    m_trt->set_run_dims(input_names[4].c_str(), m_point_labels_dims);

    // Prepare mask input (zeros)
    m_mask_input_dims = {batch_size, m_mask_input_dims[1], m_mask_input_dims[2], m_mask_input_dims[3]};
    checkRuntime(
        cudaMemsetAsync(m_mask_input_buffer.gpu(), 0, vectorProduct(m_mask_input_dims) * sizeof(float), m_stream));
    m_trt->set_run_dims(input_names[5].c_str(), m_mask_input_dims);

    // Prepare has_mask_input (0 for no mask input)
    checkRuntime(
        cudaMemsetAsync(m_has_mask_input_buffer.gpu(), 0, vectorProduct(m_has_mask_input_dims) * sizeof(float), m_stream
        ));

    // infer
    const std::map<std::string, void*> bindings = {
        {input_names[0].c_str(), image_embedding_tensor},
        {input_names[1].c_str(), high_res_feats0_tensor},
        {input_names[2].c_str(), high_res_feats1_tensor},
        {input_names[3].c_str(), m_point_coords_buffer.gpu()},
        {input_names[4].c_str(), m_point_labels_buffer.gpu()},
        {input_names[5].c_str(), m_mask_input_buffer.gpu()},
        {input_names[6].c_str(), m_has_mask_input_buffer.gpu()},
        {output_names[0].c_str(), m_low_res_masks_buffer.gpu()},
        {output_names[1].c_str(), m_iou_predictions_buffer.gpu()},
    };
    if (!m_trt->forward(bindings, m_stream)) {
        ERROR("Failed to tensorRT forward");
        throw std::runtime_error("TensorRT forward failed");
    }

    // copy back
    m_iou_predictions_dims = {batch_size, m_iou_predictions_dims[1]};
    m_low_res_masks_dims = {batch_size, m_low_res_masks_dims[1], m_low_res_masks_dims[2], m_low_res_masks_dims[3]};
    checkRuntime(
        cudaMemcpyAsync(m_iou_predictions_buffer.cpu(), m_iou_predictions_buffer.gpu(), vectorProduct(
            m_iou_predictions_dims) * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_low_res_masks_buffer.cpu(), m_low_res_masks_buffer.gpu(), vectorProduct(m_low_res_masks_dims)
            * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(cudaStreamSynchronize(m_stream));

    // postprocess
    int low_res_num_masks = m_low_res_masks_dims[1];
    int low_res_height = m_low_res_masks_dims[2];
    int low_res_width = m_low_res_masks_dims[3];
    Results results;
    for (int b = 0; b < batch_size; ++b) {
        Result result;
        for (int n = 0; n < low_res_num_masks; ++n) {
            float* src_ptr = m_low_res_masks_buffer.cpu() + b * low_res_num_masks * low_res_height * low_res_width + n *
                             low_res_height * low_res_width;
            cv::Mat mask(low_res_height, low_res_width, CV_32FC1, src_ptr);
            // 放大后的mask
            auto mask_f32 = mask_postprocessing(mask, original_image_size);
            cv::Mat mask_u8;
            cv::Mat threshold_mask = mask_f32 > mask_threshold_; // CV_8UC1, 0 或 1
            threshold_mask.convertTo(mask_u8, CV_8UC1, 255.0); // 转为 0 或 1
            result.masks.push_back(mask_u8);
            // 低分辨率mask
            cv::Mat low_res_mask(low_res_height, low_res_width, CV_32FC1);
            std::memcpy(low_res_mask.data, src_ptr, low_res_height * low_res_width * sizeof(float));
            result.low_res_masks.push_back(low_res_mask);
            // IoU预测
            result.scores.push_back(m_iou_predictions_buffer.cpu()[b * low_res_num_masks + n]);
        }
        // sort masks by scores
        std::vector<int> indices(low_res_num_masks);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) {
            return result.scores[i1] > result.scores[i2];
        });
        // reorder masks and scores
        std::vector<cv::Mat> sorted_masks;
        std::vector<cv::Mat> sorted_low_res_masks;
        std::vector<float> sorted_scores;
        for (int idx: indices) {
            sorted_masks.push_back(result.masks[idx]);
            sorted_low_res_masks.push_back(result.low_res_masks[idx]);
            sorted_scores.push_back(result.scores[idx]);
        }
        result.masks = std::move(sorted_masks);
        result.low_res_masks = std::move(sorted_low_res_masks);
        result.scores = std::move(sorted_scores);

        results.emplace_back(result);
    }

    return results;
}

Sam2::Results Sam2::Decoder::forward_logits(const cv::Size& original_image_size, const ImageEmbedding& image_embedding,
                                            const PromptPoints& prompt_points) {
    // memory
    int batch_size = prompt_points.size();
    if (batch_size <= 0) return {};
    adjust_memory(batch_size, 2);

    // image embedding tensor
    float* image_embedding_tensor = image_embedding.image_embedding_;
    float* high_res_feats0_tensor = image_embedding.high_res_feats0_;
    float* high_res_feats1_tensor = image_embedding.high_res_feats1_;

    // point coords and labels tensors
    int index = 0;
    for (const auto& point: prompt_points) {
        auto transformed = ImagePreprocessor::apply_coords({point.x, point.y}, original_image_size, target_size_);
        m_point_coords_buffer.cpu()[index * 2] = transformed.x;
        m_point_coords_buffer.cpu()[index * 2 + 1] = transformed.y;
        m_point_labels_buffer.cpu()[index] = point.label == 1 ? 1.0f : 0.0f;
        index++;
        // 在torch sam2官方中，SAM会默认添加一个补齐点，坐标为(0,0)，标签为-1，表示这个点不参与计算。这里我们也做同样的处理。
        m_point_coords_buffer.cpu()[index * 2] = 0.0f;
        m_point_coords_buffer.cpu()[index * 2 + 1] = 0.0f;
        m_point_labels_buffer.cpu()[index] = -1.0f; // -1表示补齐点
        index++;
    }
    m_point_coords_dims = {batch_size, 2, 2};
    m_point_labels_dims = {batch_size, 2};
    checkRuntime(
        cudaMemcpyAsync(m_point_coords_buffer.gpu(), m_point_coords_buffer.cpu(), vectorProduct(m_point_coords_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_point_labels_buffer.gpu(), m_point_labels_buffer.cpu(), vectorProduct(m_point_labels_dims) *
            sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_trt->set_run_dims(input_names[3].c_str(), m_point_coords_dims);
    m_trt->set_run_dims(input_names[4].c_str(), m_point_labels_dims);

    // Prepare mask input (zeros)
    m_mask_input_dims = {batch_size, m_mask_input_dims[1], m_mask_input_dims[2], m_mask_input_dims[3]};
    checkRuntime(
        cudaMemsetAsync(m_mask_input_buffer.gpu(), 0, vectorProduct(m_mask_input_dims) * sizeof(float), m_stream));
    m_trt->set_run_dims(input_names[5].c_str(), m_mask_input_dims);

    // Prepare has_mask_input (0 for no mask input)
    checkRuntime(
        cudaMemsetAsync(m_has_mask_input_buffer.gpu(), 0, vectorProduct(m_has_mask_input_dims) * sizeof(float), m_stream
        ));

    // infer
    const std::map<std::string, void*> bindings = {
        {input_names[0].c_str(), image_embedding_tensor},
        {input_names[1].c_str(), high_res_feats0_tensor},
        {input_names[2].c_str(), high_res_feats1_tensor},
        {input_names[3].c_str(), m_point_coords_buffer.gpu()},
        {input_names[4].c_str(), m_point_labels_buffer.gpu()},
        {input_names[5].c_str(), m_mask_input_buffer.gpu()},
        {input_names[6].c_str(), m_has_mask_input_buffer.gpu()},
        {output_names[0].c_str(), m_low_res_masks_buffer.gpu()},
        {output_names[1].c_str(), m_iou_predictions_buffer.gpu()},
    };
    if (!m_trt->forward(bindings, m_stream)) {
        ERROR("Failed to tensorRT forward");
        throw std::runtime_error("TensorRT forward failed");
    }

    // copy back
    m_iou_predictions_dims = {batch_size, m_iou_predictions_dims[1]};
    m_low_res_masks_dims = {batch_size, m_low_res_masks_dims[1], m_low_res_masks_dims[2], m_low_res_masks_dims[3]};
    checkRuntime(
        cudaMemcpyAsync(m_iou_predictions_buffer.cpu(), m_iou_predictions_buffer.gpu(), vectorProduct(
            m_iou_predictions_dims) * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(
        cudaMemcpyAsync(m_low_res_masks_buffer.cpu(), m_low_res_masks_buffer.gpu(), vectorProduct(m_low_res_masks_dims)
            * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    checkRuntime(cudaStreamSynchronize(m_stream));

    // postprocess
    int low_res_num_masks = m_low_res_masks_dims[1];
    int low_res_height = m_low_res_masks_dims[2];
    int low_res_width = m_low_res_masks_dims[3];
    Results results;
    for (int b = 0; b < batch_size; ++b) {
        Result result;
        for (int n = 0; n < low_res_num_masks; ++n) {
            float* src_ptr = m_low_res_masks_buffer.cpu() + b * low_res_num_masks * low_res_height * low_res_width + n *
                low_res_height * low_res_width;
            cv::Mat mask(low_res_height, low_res_width, CV_32FC1, src_ptr);
            // 放大后的mask
            cv::Mat mask_f32;
            cv::resize(mask, mask_f32, original_image_size, 0, 0, cv::INTER_LINEAR);
            result.masks.push_back(mask_f32);
            // IoU预测
            result.scores.push_back(m_iou_predictions_buffer.cpu()[b * low_res_num_masks + n]);
        }
        // sort masks by scores
        std::vector<int> indices(low_res_num_masks);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) {
            return result.scores[i1] > result.scores[i2];
        });
        // reorder masks and scores
        std::vector<cv::Mat> sorted_masks;
        std::vector<cv::Mat> sorted_low_res_masks;
        std::vector<float> sorted_scores;
        for (int idx: indices) {
            sorted_masks.push_back(result.masks[idx]);
            sorted_scores.push_back(result.scores[idx]);
        }
        result.masks = std::move(sorted_masks);
        result.scores = std::move(sorted_scores);

        results.emplace_back(result);
    }

    return results;
}

void Sam2::Decoder::adjust_memory(int batch_size, int num_points) {
    if (batch_size <= 0) {
        ERROR("batch_size should be greater than 0");
        return;
    }
    // cpu
    m_point_coords_buffer.cpu(batch_size * (num_points * 2));
    m_point_labels_buffer.cpu(batch_size * num_points);
    m_mask_input_buffer.cpu(batch_size * m_mask_input_dims[1] * m_mask_input_dims[2] * m_mask_input_dims[3]);
    m_has_mask_input_buffer.cpu(1);
    m_iou_predictions_buffer.cpu(batch_size * m_iou_predictions_dims[1]);
    m_low_res_masks_buffer.
            cpu(batch_size * m_low_res_masks_dims[1] * m_low_res_masks_dims[2] * m_low_res_masks_dims[3]);

    // gpu
    m_point_coords_buffer.gpu(batch_size * (num_points * 2));
    m_point_labels_buffer.gpu(batch_size * num_points);
    m_mask_input_buffer.gpu(batch_size * m_mask_input_dims[1] * m_mask_input_dims[2] * m_mask_input_dims[3]);
    m_has_mask_input_buffer.gpu(1);
    m_iou_predictions_buffer.gpu(batch_size * m_iou_predictions_dims[1]);
    m_low_res_masks_buffer.gpu(
        batch_size * m_low_res_masks_dims[1] * m_low_res_masks_dims[2] * m_low_res_masks_dims[3]);
}

cv::Mat Sam2::Decoder::mask_postprocessing(cv::Mat& mask, const cv::Size& original_image_size) const {
    cv::resize(mask, mask, original_image_size, 0, 0, cv::INTER_LINEAR);
    // 阈值比较
    cv::Mat mask_u8;
    cv::Mat threshold_mask = mask > mask_threshold_; // CV_8UC1, 0 或 1
    threshold_mask.convertTo(mask_u8, CV_8UC1, 255.0);
    return mask_u8;
}
