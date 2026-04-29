/**
 * @author mpj
 * @date 2023/7/18 9:13
 * @version V1.0
 */
#include <NvInfer.h>
#include <cstdarg>

#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "infer.h"

namespace trt {
    using namespace std;
    using namespace nvinfer1;

    static std::string get_file_name(const std::string& path, bool include_suffix) {
        if (path.empty()) return "";

        int p = path.rfind('/');
        int e = path.rfind('\\');
        p = std::max(p, e);
        p += 1;

        // include suffix
        if (include_suffix) return path.substr(p);

        int u = path.rfind('.');
        if (u == -1) return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    void _log_func(const char* file, int line, const char* prefix, const char* fmt, ...) {
        // 这是一个可变参数函数，需要使用va_list来获取参数
        va_list args;
        // 初始化参数列表
        va_start(args, fmt);
        char buffer[2048];
        // 文件名
        std::string filename = get_file_name(file, true);
        // 将格式化的字符串写入buffer中
        int n = snprintf(buffer, sizeof(buffer), "[%s:%d]\t%s: ", filename.c_str(), line, prefix);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, args);
        va_end(args);
        // 将buffer中的内容写入到stdout中
        fprintf(stdout, "%s\n", buffer);
    }

    static string get_dtype_name(DType dtype) {
        switch (dtype) {
        case DType::FLOAT:
            return "FLOAT";
        case DType::HALF:
            return "HALF";
        case DType::INT8:
            return "INT8";
        case DType::INT32:
            return "INT32";
        case DType::BOOL:
            return "BOOL";
        case DType::UINT8:
            return "UINT8";
        default:
            return "UNKNOWN";
        }
    }

    static string get_format_shape(const Dims& shape) {
        stringstream output;
        char buffer[64];
        const char* fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.nbDims; ++i) {
            // 如果是第一个维度，那么使用%d，否则使用x%d
            snprintf(buffer, sizeof(buffer), fmts[i != 0], shape.d[i]);
            output << buffer;
        }
        return output.str();
    }

    Timer::Timer() {
        checkRuntime(cudaEventCreate((cudaEvent_t*)&m_start));
        checkRuntime(cudaEventCreate((cudaEvent_t*)&m_stop));
    }

    Timer::~Timer() {
        checkRuntime(cudaEventDestroy((cudaEvent_t)m_start));
        checkRuntime(cudaEventDestroy((cudaEvent_t)m_stop));
    }

    void Timer::start(void* stream) {
        m_stream = stream;
        checkRuntime(cudaEventRecord((cudaEvent_t)m_start, (cudaStream_t)m_stream));
    }

    float Timer::stop(const char* prefix, bool print) {
        checkRuntime(cudaEventRecord((cudaEvent_t)m_stop, (cudaStream_t)m_stream));
        checkRuntime(cudaEventSynchronize((cudaEvent_t)m_stop));

        float ms = 0;
        checkRuntime(cudaEventElapsedTime(&ms, (cudaEvent_t)m_start, (cudaEvent_t)m_stop));

        if (print) INFO("%s: %.3f ms", prefix, ms);
        return ms;
    }


    BaseMemory::BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes) {
        reference(cpu, cpu_bytes, gpu, gpu_bytes);
    }

    BaseMemory::~BaseMemory() {
        release();
    }

    void BaseMemory::reference(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes) {
        // 释放之前的内存
        release();

        // 先把内存的指针和大小清除
        if (cpu == nullptr || cpu_bytes == 0) {
            m_cpu = nullptr;
            m_cpu_bytes = 0;
        }

        if (gpu == nullptr || gpu_bytes == 0) {
            m_gpu = nullptr;
            m_gpu_bytes = 0;
        }

        // 引用新的内存
        this->m_cpu = cpu;
        this->m_cpu_bytes = cpu_bytes;
        this->m_cpu_capacity = cpu_bytes;
        this->m_gpu = gpu;
        this->m_gpu_bytes = gpu_bytes;
        this->m_gpu_capacity = gpu_bytes;

        this->m_owner_cpu = !(cpu && cpu_bytes > 0);
        this->m_owner_gpu = !(gpu && gpu_bytes > 0);
    }

    void* BaseMemory::gpu_realloc(size_t bytes) {
        if (m_gpu_capacity < bytes) {
            release_gpu();

            // 重新分配内存
            m_gpu_capacity = bytes;
            checkRuntime(cudaMalloc(&m_gpu, m_gpu_capacity));
        }
        m_gpu_bytes = bytes;
        return m_gpu;
    }

    void* BaseMemory::cpu_realloc(size_t bytes) {
        if (m_cpu_capacity < bytes) {
            release_cpu();

            // 重新分配内存
            m_cpu_capacity = bytes;
            checkRuntime(cudaMallocHost(&m_cpu, m_cpu_capacity));
            Assertf(m_cpu != nullptr, "cudaMallocHost failed");
        }
        m_cpu_bytes = bytes;
        return m_cpu;
    }

    void BaseMemory::release_gpu() {
        if (m_gpu) {
            if (m_owner_gpu) {
                checkRuntime(cudaFree(m_gpu));
            }
            m_gpu = nullptr;
        }
        m_gpu_bytes = 0;
        m_gpu_capacity = 0;
    }

    void BaseMemory::release_cpu() {
        if (m_cpu) {
            if (m_owner_cpu) {
                checkRuntime(cudaFreeHost(m_cpu));
            }
            m_cpu = nullptr;
        }
        m_cpu_bytes = 0;
        m_cpu_capacity = 0;
    }

    void BaseMemory::release() {
        release_gpu();
        release_cpu();
    }

    static string dtype2string(DType dtype) {
        switch (dtype) {
        case DType::FLOAT:
            return "FLOAT";
        case DType::HALF:
            return "HALF";
        case DType::INT8:
            return "INT8";
        case DType::INT32:
            return "INT32";
        case DType::BOOL:
            return "BOOL";
        case DType::UINT8:
            return "UINT8";
        case DType::FP8:
            return "FP8";
        case DType::BF16:
            return "BF16";
        case DType::INT64:
            return "INT64";
        case DType::INT4:
            return "INT4";
        }
    }

    // 用于NVInfer的日志输出
    class _native_nvinfer_logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const AsciiChar* msg) noexcept override {
            switch (severity) {
            case Severity::kINTERNAL_ERROR:
                ERROR("NVInfer INTERNAL_ERROR: %s", msg);
                break;
            case Severity::kERROR:
                ERROR("NVInfer ERROR: %s", msg);
                break;
                //						case Severity::kWARNING:
                //							WARN("NVInfer WARNING: %s", msg);
                //							break;
                //						case Severity::kINFO:
                //							INFO("NVInfer INFO: %s", msg);
                //							break;
                //						default:
                //							break;
            }
        }
    };

    // 全局的NVInfer日志输出
    static _native_nvinfer_logger g_logger;

    template <typename T>
    static void destroy_nvidia_pointer(T* ptr) {
        delete ptr;
    }

    // 读入trt的文件
    static vector<uint8_t> load_file(const string& file) {
        ifstream in(file, ios::in | ios::binary);
        if (!in) {
            ERROR("Cannot open %s", file.c_str());
            return {};
        }

        // 获取文件大小
        in.seekg(0, ios::end);
        size_t length = in.tellg();

        vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, ios::beg);
            data.resize(length);
            in.read((char*)data.data(), length);
        }
        in.close();
        return data;
    }

    class _native_engine_context {
    public:
        virtual ~_native_engine_context() { destroy(); }

        // 构建上下文、引擎和运行时
        bool construct(const void* pdata, size_t size) {
            destroy();

            if (pdata == nullptr || size == 0) return false;

            // 构建运行时
            m_runtime = shared_ptr<IRuntime>(createInferRuntime(g_logger),
                                             destroy_nvidia_pointer<IRuntime>);
            if (m_runtime == nullptr) return false;

            // 构建引擎
            m_engine = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(pdata, size),
                                               destroy_nvidia_pointer<ICudaEngine>);
            if (m_engine == nullptr) return false;

            // 构建上下文
            m_context = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(),
                                                      destroy_nvidia_pointer<IExecutionContext>);
            if (m_context == nullptr) return false;

            return true;
        }

    private:
        void destroy() {
            m_context.reset();
            m_engine.reset();
            m_runtime.reset();
        }

    public:
        shared_ptr<IExecutionContext> m_context;
        shared_ptr<ICudaEngine> m_engine;
        shared_ptr<IRuntime> m_runtime = nullptr;
    };

    class InferImpl : public Infer {
    public:
        shared_ptr<_native_engine_context> m_context;
        unordered_map<string, int> m_binding_name2index;
        vector<string> m_input_names;
        vector<string> m_output_names;

    public:
        virtual ~InferImpl() = default;

        bool forward(const std::map<std::string, void*>& bindings, void* stream) override {
            if (m_context == nullptr) return false;
            // 输入和输出
            for (const auto& binding : bindings) {
                auto name = binding.first;
                auto ptr = binding.second;
                if (ptr == nullptr) {
                    ERROR("Binding %s is nullptr", name.c_str());
                    return false;
                }
                if (is_input(name.c_str())) {
                    this->m_context->m_context->setInputTensorAddress(name.c_str(), ptr);
                }
                else if (is_output(name.c_str())) {
                    this->m_context->m_context->setOutputTensorAddress(name.c_str(), ptr);
                }
                else {
                    ERROR("Unknown binding name: %s", name.c_str());
                    return false;
                }
            }

            return this->m_context->m_context->enqueueV3((cudaStream_t)stream);
        }

        vector<int> run_dims(char const* tensorName) override {
            auto dims = m_context->m_context->getTensorShape(tensorName);
            return {dims.d, dims.d + dims.nbDims};
        }

        vector<int> static_dims(char const* tensorName) override {
            auto dims = m_context->m_engine->getTensorShape(tensorName);
            return {dims.d, dims.d + dims.nbDims};
        }

        int numel(char const* tensorName) override {
            auto dims = m_context->m_engine->getTensorShape(tensorName);
            return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
        }

        bool is_input(char const* tensor_name) override {
            return this->m_context->m_engine->getTensorIOMode(tensor_name) == TensorIOMode::kINPUT;
        }

        bool is_output(const char* tensorName) override {
            return this->m_context->m_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT;
        }

        std::vector<std::string> get_input_names() override {
            return m_input_names;
        }

        std::vector<std::string> get_output_names() override {
            return m_output_names;
        }

        bool set_run_dims(char const* tensorName, const vector<int>& dims) override {
            // 判断是否为输入
            if (!this->is_input(tensorName)) return false;
            Dims d;
            for (int i = 0; i < dims.size(); ++i) {
                d.d[i] = dims[i];
            }
            d.nbDims = (int)dims.size();
            return this->m_context->m_context->setInputShape(tensorName, d);
        }

        DType dtype(char const* tensorName) override {
            return (DType)this->m_context->m_engine->getTensorDataType(tensorName);
        }

        bool has_dynamic_dim() override {
            for (const auto& name : m_input_names) {
                auto dims = this->m_context->m_engine->getTensorShape(name.c_str());
                for (int i = 0; i < dims.nbDims; ++i) {
                    if (dims.d[i] == -1) return true;
                }
            }
            for (const auto& name : m_output_names) {
                auto dims = this->m_context->m_engine->getTensorShape(name.c_str());
                for (int i = 0; i < dims.nbDims; ++i) {
                    if (dims.d[i] == -1) return true;
                }
            }
            return false;
        }

        void print() override {
            INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

            INFO("Input: %d", m_input_names.size());
            for (int i = 0; i < m_input_names.size(); ++i) {
                auto name = m_input_names[i];
                auto dtype = this->dtype(name.c_str());
                auto dims = this->m_context->m_engine->getTensorShape(name.c_str());
                INFO("\t%d.%s : type %s shape {%s}", i, name.c_str(), dtype2string(dtype).c_str(),
                     get_format_shape(dims).c_str());
            }

            INFO("Output: %d", m_output_names.size());
            for (int i = 0; i < m_output_names.size(); ++i) {
                auto name = m_output_names[i];
                auto dtype = this->dtype(name.c_str());
                auto dims = this->m_context->m_engine->getTensorShape(name.c_str());
                INFO("\t%d.%s : type %s shape {%s}", i, name.c_str(), dtype2string(dtype).c_str(),
                     get_format_shape(dims).c_str());
            }
        }

        void setup() {
            auto engine = this->m_context->m_engine;
            int nbBindings = engine->getNbIOTensors();

            m_binding_name2index.clear();
            for (int i = 0; i < nbBindings; ++i) {
                auto name = engine->getIOTensorName(i);
                m_binding_name2index[name] = i;
                if (this->is_input(name))
                    m_input_names.emplace_back(name);
                else if (this->is_output(name))
                    m_output_names.emplace_back(name);
                else
                    ERROR("Unknown binding name: %s", name);
            }
        }

        bool load(const string& file) {
            auto data = load_file(file);
            if (data.empty()) {
                ERROR("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
                return false;
            }
            return this->construct(data.data(), data.size());
        }

        bool construct(const void* data, size_t size) {
            m_context = make_shared<_native_engine_context>();
            if (!m_context->construct(data, size)) {
                ERROR("Cannot construct engine context");
                return false;
            }

            setup();
            return true;
        }
    };

    Infer* loadRaw(const std::string& file) {
        auto* impl = new InferImpl();
        if (!impl->load(file)) {
            delete impl;
            return nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load(const string& file) {
        return std::shared_ptr<InferImpl>((InferImpl*)loadRaw(file));
    }

    std::string format_shape(const vector<int>& shape) {
        stringstream output;
        char buffer[64];
        const char* fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.size(); ++i) {
            // 如果是第一个维度，那么使用%d，否则使用x%d
            snprintf(buffer, sizeof(buffer), fmts[i != 0], shape[i]);
            output << buffer;
        }
        return output.str();
    }
} // namespace trt