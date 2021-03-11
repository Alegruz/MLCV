#pragma once

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace mlcv
{
	typedef double pixel_t;
	constexpr const size_t CIFAR100_PIXEL_SIZE = 3072;
	constexpr const size_t CIFAR100_IMAGE_SIZE = 32;
	constexpr const size_t RED_INDEX = 0;
	constexpr const size_t GREEN_INDEX = 1;
	constexpr const size_t BLUE_INDEX = 2;
	constexpr const size_t RGB = 3;

	enum class eImageOrientation
	{
		STRAIGHT,
		RGB
	};

	inline pixel_t* ConvertToLodePngCompatibleArrayMalloc(uint8_t* image, size_t size, eImageOrientation imageOrientation)
	{
		if (image != nullptr)
		{
			if (imageOrientation == eImageOrientation::STRAIGHT)
				{
				pixel_t* newPixelByDouble = new pixel_t[CIFAR100_PIXEL_SIZE];
				uint8_t newPixelByUint8[CIFAR100_PIXEL_SIZE] = { 0, };
				for (size_t i = 0; i < CIFAR100_PIXEL_SIZE; ++i)
				{
					newPixelByDouble[3 * (i % 1024) + (i / 1024)] = static_cast<double>(image[i]) / 255.0;
					newPixelByUint8[3 * (i % 1024) + (i / 1024)] = image[i];
				}

				memcpy(image, newPixelByUint8, CIFAR100_PIXEL_SIZE);

				return newPixelByDouble;
			}
		}

		return nullptr;
	}

	inline void ConvertToLodePngCompatibleArray(uint8_t* image, size_t size, eImageOrientation imageOrientation)
	{
		if (image != nullptr)
		{
			if (imageOrientation == eImageOrientation::STRAIGHT)
				{
				uint8_t newPixel[CIFAR100_PIXEL_SIZE] = { 0, };
				for (size_t i = 0; i < CIFAR100_PIXEL_SIZE; ++i)
				{
					newPixel[3 * (i % 1024) + (i / 1024)] = image[i];
				}

				memcpy(image, newPixel, CIFAR100_PIXEL_SIZE);
			}
		}
	}

	inline void SwapUint8(uint8_t& left, uint8_t& right)
	{
		if (left != right)
		{
			left = (left & right) + (left | right);
			right = left + (~right) + 1;
			left = left + (~right) + 1;
		}
	}
}