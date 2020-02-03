#include "Slices.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <minwindef.h>

template<class T> __device__ float Multiplier() { return 1.0f; }
template<> __device__ float Multiplier<uchar>() { return 255.0f; }
template<> __device__ float Multiplier<schar>() { return 127.0f; }
template<> __device__ float Multiplier<ushort>() { return 65535.0f; }
template<> __device__ float Multiplier<short>() { return 32767.0f; }

template<class T>
__global__ void RGBToSlices32(float* target, const T* source, uint pitch, uint width, uint sliceSize)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	float coef = 1.0f / Multiplier<T>();
	uint pitchIdx = y * pitch + x;
	uint rgbIdx = (y * width + x) * 3;

	target[pitchIdx] = coef * source[rgbIdx];
	target[pitchIdx + sliceSize] = coef * source[rgbIdx + 1];
	target[pitchIdx + 2 * sliceSize] = coef * source[rgbIdx + 2];
}

template<class T>
auto Slices::RGBHostToSliceDevice(const T* host, uint width, uint height, uint depth) -> cudaPitchedPtr
{
	T* sourceRGB_dev{};
	CuInit(sourceRGB_dev, host);

	cudaPitchedPtr target_dev{};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	CheckCudaErrors(cudaMalloc3D(&target_dev, extent));

	const size_t pitchedBytesPerSlice = target_dev.pitch * target_dev.ysize;
	const size_t sliceSize = pitchedBytesPerSlice / sizeof(float);

	dim3 blockSize(min(PowTwoDivider(Width), 16), min(PowTwoDivider(Height), 16));
	dim3 gridSize(Width / blockSize.x, Height / blockSize.y);

	RGBToSlices32 << <gridSize, blockSize >> > ((float*)target_dev.ptr, sourceRGB_dev, target_dev.pitch, width, sliceSize);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaFree(sourceRGB_dev));
	return target_dev;
}

template<class T>
__global__ void Slices32ToRGB(float* source_dev, T* target, uint pitch, uint width, uint sliceSize)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	float coef = Multiplier<T>();
	uint pitchedIdx = y * pitch + x;
	uint rgbIdx = (y * width + x) * 3;

	target[rgbIdx] = coef * source_dev[pitchedIdx];
	target[rgbIdx + 1] = coef * source_dev[pitchedIdx + sliceSize];
	target[rgbIdx + 2] = coef * source_dev[pitchedIdx + 2 * sliceSize];
}

/**
 * \brief pitched sliced float32 -> uchar8 on device -> uchar8 on host.
 */
template<class T>
auto Slices::SliceDeviceToRGBHost(T* target_host, cudaPitchedPtr source_dev) -> void
{
	auto sizeBytes = GetSize() * sizeof(T);
	T* temp8bits_dev{}; // 1D temporary device array 8bits RGB
	CuInit(temp8bits_dev);

	const size_t pitchedBytesPerSlice = source_dev.pitch * source_dev.ysize;
	const size_t sliceSize = pitchedBytesPerSlice / sizeof(float);

	dim3 blockSize(min(PowTwoDivider(Width), 16), min(PowTwoDivider(Height), 16));
	dim3 gridSize(Width / blockSize.x, Height / blockSize.y);

	Slices32ToRGB << <gridSize, blockSize >> > ((float*)source_dev.ptr, temp8bits_dev, source_dev.pitch, Width, sliceSize);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaMemcpy(target_host, temp8bits_dev, sizeBytes, cudaMemcpyDeviceToHost));
	CheckCudaErrors(cudaFree(temp8bits_dev));
}

/**
 * \brief Allocate memory and copy host array to device array.
 * \tparam T typically uint8 (unsigned char)
 * \param deviceArray result array
 * \param hostArray origin array
 */
template<class T>
auto Slices::CuInit(T*& deviceArray, const T* hostArray) -> void
{
	auto sizeBytes = GetSize() * sizeof(T);
	CuInit(deviceArray);
	CheckCudaErrors(cudaMemcpy(deviceArray, hostArray, sizeBytes, cudaMemcpyHostToDevice));
}

template<class T>
auto Slices::CuInit(T*& deviceArray) -> void
{
	if (GetSize() == 0) return;
	if (deviceArray != nullptr)
	{
		CheckCudaErrors(cudaFree(deviceArray));
		deviceArray = nullptr;
	}
	auto sizeBytes = GetSize() * sizeof(T);
	CheckCudaErrors(cudaMalloc(reinterpret_cast<void**>(&deviceArray), sizeBytes));
	CheckCudaErrors(cudaMemset(deviceArray, 0, sizeBytes));
}

template<class T>
auto Slices::CPUInit(T*& array) -> void
{
	if (GetSize() == 0)
		return;
	if (array != nullptr)
		free(array);
	auto sizeBytes = GetSize() * sizeof(T);
	array = static_cast<T*>(malloc(sizeBytes));
	memset(array, 0, GetSize());
}

Slices::Slices(MessageChangedCallback func) :
	LogMessageChangedCallback(func)
{}

auto Slices::Test_RGBSliceRGB() -> void
{
	CPUInit(OriIm8);
	auto OriImageFP32_dev = RGBHostToSliceDevice(OriIm8, Width, Height, Colors);
	CPUInit(ResultIm8);
	SliceDeviceToRGBHost(ResultIm8, OriImageFP32_dev);
}

int main()
{
	Slices sut([](const wchar_t* arg) {wprintf(arg); });
	sut.Test_RGBSliceRGB();
}