// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C"
{
#include "libdivide.h"
}


//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
template<typename A = std::milli>
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

__device__ __forceinline__ uint32_t cuda_mullhi_u32(uint32_t x, uint32_t y) {
    uint64_t xl = x, yl = y;
    uint64_t rl = xl * yl;
    return (uint32_t)(rl >> 32);
}
__device__ __forceinline__ uint32_t fastDivide(uint32_t x, uint32_t a, uint32_t b, uint32_t m)
{
    uint64_t c = b;
    asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(c) : "r"(a), "r"(x));
    return (c >> (32 + m));
}
//-----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cuda_u32_branchfree_do(uint32_t numer, const struct libdivide_u32_branchfree_t *denom) {
    const uint32_t q = __umulhi(denom->magic, numer);
    const uint32_t t = ((numer - q) >> 1) + q;
    return t >> denom->more;
}
//-----------------------------------------------------------------------------
__global__ void validateFastDivide(const uint32_t *d_D, const uint32_t *d_A, const uint32_t *d_B, const uint32_t *d_M,
                                   unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t x = (id * threadStep) + 2;
    
        const uint32_t a = d_A[id];
        const uint32_t b = d_B[id];
        const uint32_t m = d_M[id];
        

        // Loop through arrays
        for(unsigned int i = 0; i < count; i++) {
            const uint32_t divide = d_D[i] / x;

            const uint32_t fastDivideRes = fastDivide(d_D[i], a, b, m);
           
            if(divide != fastDivideRes) {
                printf("FastDivide failed for %u / %u = %u (correct answer %u)\n\ta=%u, b=%u, m=%u\n", d_D[i], x, fastDivideRes, divide, d_A[i], d_B[i], d_M[i]);
            }
        }
    }
}
//-----------------------------------------------------------------------------
__global__ void validateLibDivide(const uint32_t *d_D, const libdivide_u32_branchfree_t *d_L,
                                   unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t x = (id * threadStep) + 2;
    
        // Loop through arrays
        for(unsigned int i = 0; i < count; i++) {
            const uint32_t divide = d_D[i] / x;

            const uint32_t fastDivideRes = cuda_u32_branchfree_do(d_D[i], d_L + id);
           
            if(divide != fastDivideRes) {
                printf("LibDivide failed for %u / %u = %u (correct answer %u)\n", d_D[i], x, fastDivideRes, divide);
            }
        }
    }
}
//-----------------------------------------------------------------------------
__global__ void testDivide(const uint32_t *d_D, uint32_t *d_R,
                           unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t x = (id * threadStep) + 2;
        
        // Loop through arrays and write output
        uint32_t r = 0;
        for(unsigned int i = 0; i < count; i++) {
            r += d_D[i] / x;
        }
        d_R[id] = r;
    }
}

//-----------------------------------------------------------------------------
__global__ void testFastDivideC(const uint32_t *d_D, const uint32_t *d_A, const uint32_t *d_B, const uint32_t *d_M, uint32_t *d_R,
                                unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t a = d_A[id];
        const uint32_t b = d_B[id];
        const uint32_t m = d_M[id];

        // Loop through arrays
        uint32_t r = 0;
        for(unsigned int i = 0; i < count; i++) {
            r += (((uint64_t)d_D[i] * a) + b) >> (32 + m);
        }
        d_R[id] = r;
    }
}
//-----------------------------------------------------------------------------
__global__ void testLibDivide(const uint32_t *d_D, const libdivide_u32_branchfree_t *d_L, uint32_t *d_R,
                              unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        // Loop through arrays
        uint32_t r = 0;
        for(unsigned int i = 0; i < count; i++) {
            r += cuda_u32_branchfree_do(d_D[i], d_L + id);
        }
        d_R[id] = r;
    }
}
//-----------------------------------------------------------------------------
__global__ void testFastDividePTX(const uint32_t *d_D, const uint32_t *d_A, const uint32_t *d_B, const uint32_t *d_M, uint32_t *d_R,
                                  unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t a = d_A[id];
        const uint32_t b = d_B[id];
        const uint32_t m = d_M[id];

        // Loop through arrays
        uint32_t r = 0;
        for(unsigned int i = 0; i < count; i++) {
            r += fastDivide(d_D[i], a, b, m);
        }
        d_R[id] = r;
    }
}
//-----------------------------------------------------------------------------
__global__ void testFloatDivide(const uint32_t *d_D, uint32_t *d_R,
                                unsigned int count, unsigned int threadStep, unsigned int numThreads)
{
    const unsigned int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if(id < numThreads) {
        const uint32_t x = (id * threadStep) + 1;

        // Loop through arrays
        uint32_t r = 0;
        for(unsigned int i = 0; i < count; i++) {
            //d_R[(id * numThreads) + i] = (((uint64_t)x * d_A[i]) + d_B[i]) >> (32 + d_M[i]);
            r += (uint32_t)__fdividef(d_D[i], x);
        }
        d_R[id] = r;
    }
}
//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
std::tuple<uint32_t, uint32_t, uint32_t> calcFastDivideConstants(uint32_t d)
{
    const uint32_t m = (uint32_t)std::floor(std::log2(d));

    const uint32_t uintMax = std::numeric_limits<uint32_t>::max();
    if(d == (1 << m)) {
        return std::make_tuple(uintMax, uintMax, m);
    }
    else {
        const uint32_t t = (1ull << (m + 32)) / d;
        const uint32_t r = ((t * d) + d) & uintMax;
        if(r <= (1 << m)) {
            return std::make_tuple(t + 1, 0, m);
        }
        else {
            return std::make_tuple(t, t, m);
        }
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        unsigned int numThreads = (argc < 2) ? 131072 : std::stoul(argv[1]);
        unsigned int numDividesPerThread = (argc < 3) ? 131072 : std::stoul(argv[2]);
    
        // Determine how coursely threads and loops will sample 
        unsigned int threadStep = (std::numeric_limits<uint32_t>::max() - 1) / numThreads;
        unsigned int loopStep = (std::numeric_limits<uint32_t>::max() - 1) / numDividesPerThread;

        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Configure input data
        //------------------------------------------------------------------------
        // Create arrays to hold divisors and coefficients
        auto d = allocateHostDevice<uint32_t>(numDividesPerThread);
        auto a = allocateHostDevice<uint32_t>(numThreads);
        auto b = allocateHostDevice<uint32_t>(numThreads);
        auto m = allocateHostDevice<uint32_t>(numThreads);
        auto l = allocateHostDevice<libdivide_u32_branchfree_t>(numThreads);

        // Allocate host array for results
        uint32_t *d_r = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_r, numThreads * sizeof(uint32_t)));

        {
            Timer<std::milli> t("Generating input data:");

            // Loop through inputs and add divisors
            for(unsigned int i = 0; i < numDividesPerThread; i++) {
                d.first[i] = i * loopStep;
            }

            for(unsigned int i = 0; i < numThreads; i++) {
                uint32_t d = (i * threadStep) + 2;

                // Calculate coefficients
                std::tie(a.first[i], b.first[i], m.first[i]) = calcFastDivideConstants(d);
                l.first[i] = libdivide_u32_branchfree_gen(d);
            }
        }
        {
            Timer<std::milli> t("Uploading:");
            hostToDeviceCopy(d, numDividesPerThread);
            hostToDeviceCopy(a, numThreads);
            hostToDeviceCopy(b, numThreads);
            hostToDeviceCopy(m, numThreads);
            hostToDeviceCopy(l, numThreads);
        }
        dim3 threads(32, 1);
        dim3 grid(((numThreads + 31) / 32), 1);


        {
            Timer<std::milli> t("Testing:");
        
            validateFastDivide<<<grid, threads>>>(d.second, a.second, b.second, m.second, 
                                                  numDividesPerThread, threadStep, numThreads);
            validateLibDivide<<<grid, threads>>>(d.second, l.second, 
                                                  numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
        {
            Timer<std::milli> t("Benchmark divide:");
            testDivide<<<grid, threads>>>(d.second, d_r,
                                          numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
        {
            Timer<std::milli> t("Benchmark fast divide C:");
            testFastDivideC<<<grid, threads>>>(d.second, a.second, b.second, m.second, d_r,
                                               numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
        {
            Timer<std::milli> t("Benchmark fast divide PTX:");
            testFastDividePTX<< <grid, threads >> > (d.second, a.second, b.second, m.second, d_r,
                                                    numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
         {
            Timer<std::milli> t("Benchmark lib divide:");
            testLibDivide<< <grid, threads >> > (d.second, l.second, d_r,
                                                 numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
        {
            Timer<std::milli> t("Benchmark float divide:");
            testFloatDivide << <grid, threads >> > (d.second, d_r,
                                                    numDividesPerThread, threadStep, numThreads);
            cudaDeviceSynchronize();
        }
    }
    catch(const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

