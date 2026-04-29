/**
 * @author mpj
 * @date 2023/7/18 9:13
 * @version V1.0
 */
#ifndef YOLO_TRT_NEW_INFER_H
#define YOLO_TRT_NEW_INFER_H

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace trt {
#define INFO(...) trt::_log_func(__FILE__, __LINE__, "INFO", __VA_ARGS__)
#define WARN(...) trt::_log_func(__FILE__, __LINE__, "WARN", __VA_ARGS__)
#define ERROR(...) trt::_log_func(__FILE__, __LINE__, "ERROR", __VA_ARGS__)

#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      ERROR("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

#define Assert(op)                 \
  do {                             \
    bool cond = !(!(op));          \
    if (!cond) {                   \
      ERROR("Assert failed, " #op); \
      abort();                     \
    }                              \
  } while (0)

#define Assertf(op, ...)                             \
  do {                                               \
    bool cond = !(!(op));                            \
    if (!cond) {                                     \
      ERROR("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  } while (0)


    // 日志函数
    void _log_func(const char* file, int line, const char* prefix, const char* fmt, ...);

    // 数据类型
    enum class DType : int {
        FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5, FP8 = 6, BF16 = 7, INT64 = 8, INT4 = 9,
    };

    // 计时器
    class Timer {
    public:
        Timer();

        virtual ~Timer();

        // 开始计时
        void start(void* stream = nullptr);

        // 停止计时
        float stop(const char* prefix = "Timer", bool print = true);

    private:
        void *m_start{}, *m_stop{};
        void* m_stream{};
    };

    // 内存基类
    class BaseMemory {
    public:
        BaseMemory() = default;

        // 构造函数
        BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);

        // 析构函数
        virtual ~BaseMemory();

        // 重新分配gpu内存
        virtual void* gpu_realloc(size_t bytes);

        // 重新分配cpu内存
        virtual void* cpu_realloc(size_t bytes);

        // 释放gpu内存
        void release_gpu();

        // 释放cpu内存
        void release_cpu();

        // 释放内存
        void release();

        // 引用
        void reference(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);

        // 是否拥有gpu内存
        inline bool owner_gpu() const { return m_owner_gpu; }

        // 是否拥有cpu内存
        inline bool owner_cpu() const { return m_owner_cpu; }

        // cpu内存字节数目
        inline size_t cpu_bytes() const { return m_cpu_bytes; }

        // gpu内存字节数目
        inline size_t gpu_bytes() const { return m_gpu_bytes; }

        // 获取gpu内存
        virtual inline void* get_gpu() const { return m_gpu; }

        // 获取cpu内存
        virtual inline void* get_cpu() const { return m_cpu; }

    private:
        void* m_cpu = nullptr;
        size_t m_cpu_bytes = 0, m_cpu_capacity = 0;
        bool m_owner_cpu = true;

        void* m_gpu = nullptr;
        size_t m_gpu_bytes = 0, m_gpu_capacity = 0;
        bool m_owner_gpu = true;
    };

    // 内存类
    template <typename DT>
    class Memory : public BaseMemory {
    public:
        Memory() = default;

        Memory(const Memory& other) = delete;

        Memory& operator=(const Memory& other) = delete;

        // 如果内存不足，重新分配内存，否则返回原内存
        virtual DT* gpu(size_t size) { return (DT*)BaseMemory::gpu_realloc(size * sizeof(DT)); }

        // 如果内存不足，重新分配内存，否则返回原内存
        virtual DT* cpu(size_t size) { return (DT*)BaseMemory::cpu_realloc(size * sizeof(DT)); }

        // 得到gpu上的数据个数
        inline size_t gpu_size() const { return BaseMemory::gpu_bytes() / sizeof(DT); }

        // 得到cpu上的数据个数
        inline size_t cpu_size() const { return BaseMemory::cpu_bytes() / sizeof(DT); }

        // 得到gpu上的数据
        inline DT* gpu() const { return (DT*)BaseMemory::get_gpu(); }

        // 得到cpu上的数据
        inline DT* cpu() const { return (DT*)BaseMemory::get_cpu(); }
    };

    // 推理类
    class Infer {
    public:
        // 进行推理
        virtual bool
        forward(const std::map<std::string, void*>& bindings, void* stream = nullptr) = 0;

        // 运行时，根据名字获取名字对应索引的维度
        virtual std::vector<int> run_dims(char const* tensorName) = 0;

        // 静态维度，根据名字获取名字对应索引的维度
        virtual std::vector<int> static_dims(char const* tensorName) = 0;

        // 计算元素个数
        virtual int numel(char const* tensorName) = 0;

        // 是否是输入
        virtual bool is_input(char const* tensorName) = 0;

        virtual bool is_output(char const* tensorName) = 0;

        virtual std::vector<std::string> get_input_names() = 0;

        virtual std::vector<std::string> get_output_names() = 0;

        // 设置运行时候的维度
        virtual bool set_run_dims(char const* tensorName, const std::vector<int>& dims) = 0;

        // 获取数据的类型
        virtual DType dtype(char const* tensorName) = 0;

        // 是否有动态维度
        virtual bool has_dynamic_dim() = 0;

        // 打印动静态维度
        virtual void print() = 0;
    };

    // 导入trt模型，创建推理对象
    std::shared_ptr<Infer> load(const std::string& file);

    // 格式化维度
    std::string format_shape(const std::vector<int>& shape);
}


#endif //YOLO_TRT_NEW_INFER_H