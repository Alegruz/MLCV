#pragma once

#include "Defs.h"

#include "Image.h"

namespace mlcv
{
	struct ActivationFunction
	{
	public:
		inline ActivationFunction()
		{
		}
		inline virtual void Activate(double* data, size_t size) = 0;
		inline virtual void ActivateImage(Image& image) = 0;
	};

	struct ReLu : public ActivationFunction
	{
	public:
		inline ReLu()
			: ActivationFunction()
		{
		}

		inline virtual void Activate(double* data, size_t size)
		{
			for (size_t i = 0; i < size; ++i)
			{
				if (data[i] <= 0.0)
				{
					data[i] = 0.0;
				}
			}
		}

		inline virtual void ActivateImage(Image& image)
		{
			for (size_t x = 0; x < image.GetWidth(); ++x)
			{
				for (size_t y = 0; y < image.GetHeight(); ++y)
				{
					for (size_t z = 0; z < image.GetChannelSize(); ++z)
					{
						if (image.GetPixel(x, y, z) <= 0.0)
						{
							image.SetPixel(x, y, z, 0.0);
						}
					}
				}
			}
		}
	};

	struct SoftMax : public ActivationFunction
	{
	public:
		inline SoftMax()
			: ActivationFunction()
		{
		}

		inline virtual void Activate(double* data, size_t size)
		{
			double sum = 0.0;

			for (size_t i = 0; i < size; ++i)
			{
				sum += exp(data[i]);
			}

			for (size_t i = 0; i < size; ++i)
			{
				data[i] = exp(data[i]) / sum;
			}
		}

		inline virtual void ActivateImage(Image& image)
		{
			double sum = 0.0;
			for (size_t x = 0; x < image.GetWidth(); ++x)
			{
				for (size_t y = 0; y < image.GetHeight(); ++y)
				{
					for (size_t z = 0; z < image.GetChannelSize(); ++z)
					{
						sum += exp(image.GetPixel(x, y, z));
					}
				}
			}

			for (size_t x = 0; x < image.GetWidth(); ++x)
			{
				for (size_t y = 0; y < image.GetHeight(); ++y)
				{
					for (size_t z = 0; z < image.GetChannelSize(); ++z)
					{
						image.SetPixel(x, y, z, exp(image.GetPixel(x, y, z)) / sum);
					}
				}
			}
		}
	};
}