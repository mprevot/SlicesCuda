#pragma once
#include <cuda_runtime.h>
#include <sstream>
#include <algorithm>
#include <iosfwd>
#include <string>
#include <algorithm>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define CheckCudaErrors(val) Check((val), #val, __FUNCTION__, __FILE__, __LINE__)

typedef void(__stdcall* MessageChangedCallback)(const wchar_t* string);
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

inline __device__ __host__ uint PowTwoDivider(uint n)
{
	if (n == 0)
		return 0;
	uint divider = 1;
	while ((n & divider) == 0)
		divider <<= 1;
	return divider;
}

class __declspec(dllexport) Slices
{
	uint Width{4096}, Height{2160}, Colors{3};
	uint GetSize() { return Width * Height * Colors; }
	
	MessageChangedCallback LogMessageChangedCallback{};
	cudaPitchedPtr OriImageFP32_dev{};
	uchar* ResultIm8{};
	uchar* OriIm8{};
	
	template <class T>
	auto RGBHostToSliceDevice(const T* host, uint width, uint height, uint depth)->cudaPitchedPtr;
	template <class T>
	auto SliceDeviceToRGBHost(T* target_host, cudaPitchedPtr source_dev) -> void;
	template <class T>
	auto CuInit(T*& deviceArray, const T* hostArray) -> void;
	template <class T>
	auto CuInit(T*& deviceArray) -> void;
	template <class T>
	auto CPUInit(T*& array) -> void;
	template <class ... T>
	auto LogMessage(T&& ... args) -> void;
	template <class T>
	auto Check(T result, const char* func, const char* caller, const char* file, int line) -> void;

public:
	Slices(MessageChangedCallback func);
	auto Test_RGBSliceRGB() -> void;
};

template<class ...T>
auto Slices::LogMessage(T&&... args) -> void
{
	if (LogMessageChangedCallback != nullptr)
	{
		wchar_t updatedMessage[4096];
		swprintf_s(updatedMessage, std::forward<T>(args)...);
		LogMessageChangedCallback(updatedMessage);
	}
}

template <class T>
auto Slices::Check(T result, const char* func, const char* caller, const char* file, int line) -> void
{
	if (result)
	{
		std::wstringstream o;
		auto f = std::string(file);
		std::replace(f.begin(), f.end(), '\\', '/');
		auto justfile = std::string(f.substr(f.find_last_of('/') + 1));
		o << caller << "(): " << func << " at " << justfile.c_str() << ":" << line << " [" << cudaGetErrorName((cudaError_t)result) << "]";
		LogMessage(L"%s\n", o.str().c_str());
	}
}