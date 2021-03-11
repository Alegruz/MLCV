#pragma once

#include "Defs.h"
#include "ActivationFunction.h"
#include "Filter.h"
#include "Image.h"

namespace mlcv
{
	struct Layer
	{
		friend class LayerManager;
		friend class ImageLayerManager;

	public:
		inline Layer(size_t inputSize, size_t outputSize)
			: Layer(inputSize, outputSize, nullptr, nullptr)
		{
		}

		inline Layer(size_t inputSize, size_t outputSize, pixel_t*&& bias)
			: Layer(inputSize, outputSize, std::move(bias), nullptr)
		{
		}

		inline Layer(size_t inputSize, size_t outputSize, ActivationFunction*&& activationFunction)
			: Layer(inputSize, outputSize, nullptr, std::move(activationFunction))
		{
		}

		inline Layer(size_t inputSize, size_t outputSize, pixel_t*&& bias, ActivationFunction*&& activationFunction)
			: mInputSize(inputSize)
			, mOutputSize(outputSize)
			, mBias(bias)
			, mActivationFunction(activationFunction)
			, mPrevLayer(nullptr)
			, mNextLayer(nullptr)
		{
			if (bias == nullptr)
			{
				mBias = new double[mOutputSize];
				for (size_t i = 0; i < mOutputSize; ++i)
				{
					mBias[i] = 0.0;
				}
			}
			bias = nullptr;
			activationFunction = nullptr;
		}

		inline virtual ~Layer()
		{
			if (mBias != nullptr)
			{
				delete[] mBias;
			}

			if (mActivationFunction != nullptr)
			{
				delete mActivationFunction;
			}
		}

		inline size_t GetInputSize() const
		{
			return mInputSize;
		}

		inline size_t GetOutputSize() const
		{
			return mOutputSize;
		}

		inline void SetBias(pixel_t*&& bias)
		{
			assert(bias != nullptr);

			if (mBias != nullptr)
			{
				delete[] bias;
			}

			mBias = bias;
			bias = nullptr;
		}

		inline void SetActivationFunction(ActivationFunction*&& activationFunction)
		{
			assert(activationFunction != nullptr);

			if (mActivationFunction != nullptr)
			{
				delete mActivationFunction;
			}

			mActivationFunction = activationFunction;
			activationFunction = nullptr;
		}

		virtual void Process() = 0;
		virtual void Activate() = 0;

	protected:
		inline virtual void sendDataToNextLayer() = 0;

		size_t mInputSize;
		size_t mOutputSize;
		pixel_t* mBias;
		ActivationFunction* mActivationFunction;
		Layer* mPrevLayer;
		Layer* mNextLayer;
	};

struct ImageLayer : public Layer
	{
	public:
		inline ImageLayer(size_t outputSize, Image*&& image)
			: ImageLayer(outputSize, std::move(image), nullptr, nullptr)
		{
		}

		inline ImageLayer(size_t outputSize, Image*&& image, pixel_t*&& bias)
			: ImageLayer(outputSize, std::move(image), std::move(bias), nullptr)
		{
		}

		inline ImageLayer(size_t outputSize, Image*&& image, ActivationFunction*&& activationFunction)
			: ImageLayer(outputSize, std::move(image), nullptr, std::move(activationFunction))
		{
		}

		inline ImageLayer(size_t outputSize, Image*&& image, pixel_t*&& bias, ActivationFunction*&& activationFunction)
			: Layer(image->GetChannelSize(), outputSize, std::move(bias), std::move(activationFunction))
			, mInputImage(std::move(image))
			, mOutputImage(new Image(mInputImage->GetWidth(), mInputImage->GetHeight(), mOutputSize))
		{
			image = nullptr;
		}

		inline ImageLayer(size_t outputSize, ImageLayer& layer, pixel_t*&& bias, ActivationFunction*&& activationFunction)
			: Layer(layer.GetOutputSize(), outputSize, std::move(bias), std::move(activationFunction))
			, mInputImage(nullptr)
			, mOutputImage(nullptr)
		{
			layer.TransportImageTo(*this);
			mOutputImage = new Image(mInputImage->GetWidth(), mInputImage->GetHeight(), mOutputSize);
		}


		inline virtual ~ImageLayer()
		{
			if (mInputImage != nullptr)
			{
				delete mInputImage;
			}

			if (mOutputImage != nullptr)
			{
				delete mOutputImage;
			}
		}

		inline uint8_t GetCoarseLabel() const
		{
			return mOutputImage->GetCoarseLabel();
		}

		inline uint8_t GetFineLabel() const
		{
			return mOutputImage->GetFineLabel();
		}

		inline virtual double* GetStraightenedDataMalloc() const
		{
			if (mOutputImage != nullptr)
			{
				double* data = new double[mOutputImage->GetWidth() * mOutputImage->GetHeight() * mOutputImage->GetChannelSize()];

				for (size_t x = 0; x < mOutputImage->GetWidth(); ++x)
				{
					for (size_t y = 0; y < mOutputImage->GetHeight(); ++y)
					{
						for (size_t z = 0; z < mOutputImage->GetChannelSize(); ++z)
						{
							data[z * mOutputImage->GetWidth() * mOutputImage->GetHeight() + x + y * mOutputImage->GetWidth()] = mOutputImage->GetPixel(x, y, z);
						}
					}
				}

				return data;
			}

			return nullptr;
		}

		inline Image& GetImage()
		{
			return *mOutputImage;
		}

		inline virtual void SetImage(Image*&& image)
		{
			assert(image != nullptr);

			if (mInputImage != nullptr)
			{
				delete mInputImage;
			}

			mInputImage = image;
			mInputSize = mInputImage->GetChannelSize();
			image = nullptr;
		}

		inline void SetActivationFunction(ActivationFunction*&& activationFunction)
		{
			assert(activationFunction != nullptr);

			if (mActivationFunction != nullptr)
			{
				delete mActivationFunction;
			}

			mActivationFunction = activationFunction;
			activationFunction = nullptr;
		}

		inline virtual void Process() = 0;

		inline void Activate()
		{
			if (mOutputImage != nullptr && mActivationFunction != nullptr)
			{
				for (size_t i = 0; i < mOutputSize; ++i)
				{
					mActivationFunction->ActivateImage(*mOutputImage);
				}
			}
		}

		inline void TransportImageTo(ImageLayer& layer)
		{
			layer.SetImage(std::move(mOutputImage));
			mOutputImage = nullptr;

			delete mInputImage;
			mInputImage = nullptr;
		}

	protected:
		inline virtual void sendDataToNextLayer()
		{
		}

		size_t mStride;
		Image* mInputImage;
		Image* mOutputImage;	
	};


	struct FullyConnectedLayer : public Layer
	{
	public:
		inline FullyConnectedLayer(size_t inputSize, size_t outputSize, double*&& input, double**&& weights, double*&& bias, ActivationFunction*&& activationFunction)
			: Layer(inputSize, outputSize, std::move(bias), std::move(activationFunction))
			, mInput(input)
			, mWeights(weights)
			, mOutput(new double[mOutputSize])
		{
			input = nullptr;
			weights = nullptr;
		}

		inline FullyConnectedLayer(size_t inputSize, size_t outputSize, ImageLayer& imageLayer, double**&& weights, double*&& bias, ActivationFunction*&& activationFunction)
			: Layer(inputSize, outputSize, std::move(bias), std::move(activationFunction))
			, mInput(imageLayer.GetStraightenedDataMalloc())
			, mWeights(weights)
			, mOutput(new double[mOutputSize])
		{
			weights = nullptr;
		}

		inline FullyConnectedLayer(size_t inputSize, size_t outputSize, FullyConnectedLayer& fcLayer, double**&& weights, double*&& bias, ActivationFunction*&& activationFunction, bool isDropoutEnabled)
			: Layer(inputSize, outputSize, std::move(bias), std::move(activationFunction))
			, mInput(fcLayer.mOutput)
			, mWeights(weights)
			, mOutput(new double[mOutputSize])
			, mbIsDropoutEnabled(isDropoutEnabled)
		{
			fcLayer.mOutput = nullptr;
			weights = nullptr;

			if (mbIsDropoutEnabled)
			{
				srand(time(NULL));
			}
		}

		inline double operator[](size_t index) const
		{
			assert(index <= mOutputSize);

			return mOutput[index];
		}

		inline virtual void Process()
		{
			std::cout << "input: " << mInputSize << std::endl;
			std::cout << "output: " << mOutputSize << std::endl;
			for (size_t i = 0; i < mInputSize; ++i)
			{
				for (size_t j = 0; j < mOutputSize; ++j)
				{
					if (mbIsDropoutEnabled)
					{
						if (rand() % 2 == 0)
						{
							mOutput[j] = mInput[i] * mWeights[i][j] + mBias[j];
						}
						else
						{
							mOutput[j] = 0.0;
						}
					}
				}
			}
		}

		inline virtual void Activate()
		{
			mActivationFunction->Activate(mOutput, mOutputSize);
		}

	protected:
		virtual void sendDataToNextLayer()
		{
		}

		double* mInput;
		double** mWeights;
		double* mOutput;
		bool mbIsDropoutEnabled;
	};

	struct ConvLayer : public ImageLayer
	{
	public:
		inline ConvLayer(size_t outputSize, Filter**&& filters)
			: ConvLayer(outputSize, std::move(filters), nullptr, nullptr, nullptr)
		{
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, Image*&& image)
			: ConvLayer(outputSize, std::move(filters), std::move(image), nullptr, nullptr)
		{
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, Image*&& image, pixel_t*&& bias)
			: ConvLayer(outputSize, std::move(filters), std::move(image), std::move(bias), nullptr)
		{
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, Image*&& image, ActivationFunction*&& activationFunction)
			: ConvLayer(outputSize, std::move(filters), std::move(image), nullptr, std::move(activationFunction))
		{
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, Image*&& image, pixel_t*&& bias, ActivationFunction*&& activationFunction)
			: ImageLayer(outputSize, std::move(image), std::move(bias), std::move(activationFunction))
			, mStride(1)
			, mFilters(filters)
		{
			filters = nullptr;
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, ImageLayer& layer, ActivationFunction*&& activationFunction)
			: ConvLayer(outputSize, std::move(filters), layer, nullptr, std::move(activationFunction))
		{
		}

		inline ConvLayer(size_t outputSize, Filter**&& filters, ImageLayer& layer, pixel_t*&& bias, ActivationFunction*&& activationFunction)
			: ImageLayer(outputSize, layer, std::move(bias), std::move(activationFunction))
			, mStride(1)
			, mFilters(filters)
		{
			filters = nullptr;
		}

		inline virtual ~ConvLayer()
		{
			for (size_t i = 0; i < mOutputSize; ++i)
			{
				delete mFilters[i];
			}
			delete[] mFilters;
		}

		inline virtual void Process()
		{
			std::cout << "input: " << mInputImage->GetWidth() << ", " << mInputImage->GetHeight() << ", " << mInputImage->GetChannelSize() << std::endl;
			std::cout << "output: " << mOutputImage->GetWidth() << ", " << mOutputImage->GetHeight() << ", " << mOutputImage->GetChannelSize() << std::endl;
			for (size_t i = 0; i < mOutputSize; ++i)
			{
				for (size_t z = 0; z < mInputImage->GetChannelSize(); ++z)
				{
					size_t xOffset = (mFilters[i]->GetWidth() - 1) / 2;
					size_t yOffset = (mFilters[i]->GetHeight() - 1) / 2;

					for (size_t x = xOffset; x < mInputImage->GetWidth() - xOffset; ++x)
					{
						for (size_t y = yOffset; y < mInputImage->GetHeight() - yOffset; ++y)
						{
							for (size_t filterX = 0; filterX < mFilters[i]->GetWidth(); ++filterX)
							{
								for (size_t filterY = 0; filterY < mFilters[i]->GetHeight(); ++filterY)
								{
									mOutputImage->SetPixel(x, y, i, mOutputImage->GetPixel(x, y, i) + mInputImage->GetPixel(x + filterX - xOffset, y + filterY - yOffset, z) * mFilters[i]->GetValue(filterX, filterY, z));
								}
							}
							mOutputImage->SetPixel(x, y, i, mOutputImage->GetPixel(x, y, z) + mBias[i]);
						}
					}
				}
			}
		}

	protected:
		size_t mStride;
		Filter** mFilters;
	};

	struct MaxPoolLayer : public ImageLayer
	{
	public:
		inline explicit MaxPoolLayer(size_t poolSize, size_t poolStride, Image*&& image)
			: MaxPoolLayer(poolSize, poolStride, std::move(image), nullptr)
		{
		}

		inline explicit MaxPoolLayer(size_t poolSize, size_t poolStride, Image*&& image, ActivationFunction*&& activationFunction)
			: ImageLayer(image->GetChannelSize(), std::move(image), nullptr, std::move(activationFunction))
			, mPoolSize(poolSize)
			, mPoolStride(poolStride)
		{
		}

		inline explicit MaxPoolLayer(size_t poolSize, size_t poolStride, ImageLayer& layer)
			: MaxPoolLayer(poolSize, poolStride, layer, nullptr)
		{
		}


		inline explicit MaxPoolLayer(size_t poolSize, size_t poolStride, ImageLayer& layer, ActivationFunction*&& activationFunction)
			: ImageLayer(layer.GetOutputSize(), layer, nullptr, std::move(activationFunction))
			, mPoolSize(poolSize)
			, mPoolStride(poolStride)
		{
		}

		inline void Process()
		{
			if (mOutputImage != nullptr)
			{
				mOutputImage->ResizeImage((mInputImage->GetWidth() - mPoolSize) / mPoolStride + 1, (mInputImage->GetHeight() - mPoolSize) / mPoolStride + 1);
			}

			std::cout << "input: " << mInputImage->GetWidth() << ", " << mInputImage->GetHeight() << ", " << mInputImage->GetChannelSize() << std::endl;
			std::cout << "output: " << mOutputImage->GetWidth() << ", " << mOutputImage->GetHeight() << ", " << mOutputImage->GetChannelSize() << std::endl;
			for (size_t inputZ = 0; inputZ < mInputSize; ++inputZ)
			{
				for (size_t z = 0; z < mOutputSize; ++z)
				{
					size_t xStrideCount = 0;
					for (size_t x = 0; x < mInputImage->GetWidth(); x += mPoolStride, ++xStrideCount)
					{
						size_t yStrideCount = 0;
						for (size_t y = 0; y < mInputImage->GetHeight(); y += mPoolStride, ++yStrideCount)
						{
							pixel_t maxPixel = mInputImage->GetPixel(x, y, z);
							for (size_t i = 0; i < mPoolSize; ++i)
							{
								for (size_t j = 0; j < mPoolSize; ++j)
								{
									if (maxPixel < mInputImage->GetPixel(x + i, y + j, z))
									{
										maxPixel = mInputImage->GetPixel(x + i, y + j, z);
									}
								}
							}
							mOutputImage->SetPixel(x - mPoolStride * xStrideCount, y - mPoolStride * yStrideCount, z, maxPixel);
						}
					}
				}
			}
		}

		inline void Activate()
		{
		}

	private:
		size_t mPoolSize;
		size_t mPoolStride;
	};
}