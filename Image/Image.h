#pragma once

#include <iostream>

#include "Defs.h"

namespace mlcv
{
	struct Image
	{
	public:
		inline explicit Image(size_t width, size_t height, size_t channelSize)
			: mChannelSize(channelSize)
			, mWidth(width)
			, mHeight(height)
			, mImage(new pixel_t**[mChannelSize])
			, mCoarseLabel(0)
			, mFineLabel(0)
		{
			for (size_t z = 0; z < mChannelSize; ++z)
			{
				mImage[z] = new pixel_t*[mWidth];
				for (size_t x = 0; x < mWidth; ++x)
				{
					mImage[z][x] = new pixel_t[mHeight];
				}
			}
		}

		inline explicit Image(size_t width, size_t height, size_t channelSize, pixel_t initialValue)
			: mChannelSize(channelSize)
			, mWidth(width)
			, mHeight(height)
			, mImage(new pixel_t**[mChannelSize])
			, mCoarseLabel(0)
			, mFineLabel(0)
		{
			for (size_t z = 0; z < mChannelSize; ++z)
			{
				mImage[z] = new pixel_t*[mWidth];
				for (size_t x = 0; x < mWidth; ++x)
				{
					mImage[z][x] = new pixel_t[mHeight];
					for (size_t y = 0; y < mHeight; ++y)
					{
						mImage[z][x][y] = initialValue;
					}
				}
			}
		}

		inline explicit Image(pixel_t* image, size_t width, size_t height, eImageOrientation imageOrientation)
			: Image(width, height, 3)
		{
			switch (imageOrientation)
			{
			case eImageOrientation::STRAIGHT:
				for (size_t i = 0; i < mWidth * mHeight * mChannelSize; ++i)
				{
					mImage[i % (mWidth * mHeight)][(i % (mWidth * mHeight)) / mHeight][(i % (mWidth * mHeight)) % mHeight] = image[i];
				}
				break;
			case eImageOrientation::RGB:
				for (size_t i = 0; i < mWidth * mHeight * mChannelSize; ++i)
				{
					mImage[i % 3][(i / 3) % 32][(i / 3) / 32] = image[i];
				}
				break;
			default:
				assert(false);
				break;
			}
		}

		inline Image(const Image& other)
		{
			if (&other != this)
			{
				mChannelSize = other.mChannelSize;
				mWidth = other.mWidth;
				mHeight = other.mHeight;
				mCoarseLabel = other.mCoarseLabel;
				mFineLabel = other.mFineLabel;

				mImage = new pixel_t**[mChannelSize];
				for (size_t z = 0; z < mChannelSize; ++z)
				{
					mImage[z] = new pixel_t*[mWidth];
					for (size_t x = 0; x < mWidth; ++x)
					{
						mImage[z][x] = new pixel_t[mHeight];
						for (size_t y = 0; y < mHeight; ++y)
						{
							mImage[z][x][y] = other.mImage[z][x][y];
						}
					}
				}
			}
		}

		inline Image(const Image*&& other)
			: Image(*other)
		{
			other = nullptr;
		}

		inline ~Image()
		{
			if (mImage != nullptr)
			{
				for (size_t z = 0; z < mChannelSize; ++z)
				{
					if (mImage[z] != nullptr)
					{
						for (size_t x = 0; x < mWidth; ++x)
						{
							if (mImage[z][x] != nullptr)
							{
								delete[] mImage[z][x];
							}
						}

						delete[] mImage[z];
					}
				}
				delete[] mImage;
			}
		}

		inline size_t GetWidth() const
		{
			return mWidth;
		}

		inline size_t GetHeight() const
		{
			return mHeight;
		}

		inline size_t GetChannelSize() const
		{
			return mChannelSize;
		}

		inline uint8_t* GetLodePngCompatibleImageMalloc() const
		{
			uint8_t* image = new uint8_t[mWidth * mHeight * mChannelSize];

			for (size_t x = 0; x < mWidth; ++x)
			{
				for (size_t y = 0; y < mHeight; ++y)
				{
					for (size_t z = 0; z < RGB; ++z)
					{
						if (mChannelSize == RGB)
						{
							image[mChannelSize * (y * mWidth + x) + z] = static_cast<uint8_t>(mImage[z][x][y] * 255.0);
						}
						else
						{
							pixel_t sum = 0;
							for (size_t channel = 0; channel < mChannelSize; ++channel)
							{
								sum += mImage[z][x][y];
							}
							image[RGB * (y * mWidth + x) + z] = static_cast<uint8_t>(((sum / static_cast<pixel_t>(mChannelSize)) / 3) * 255.0);
						}
					}
				}
			}

			return image;
		}

		inline double GetPixel(size_t x, size_t y, size_t z)
		{
			assert(x < mWidth && y < mHeight && z < mChannelSize);

			return mImage[z][x][y];
		}

		inline const pixel_t* const * GetImageByChannel(size_t index) const
		{
			assert(index <= mChannelSize);

			return mImage[index];
		}

		inline void SetPixel(size_t x, size_t y, size_t z, pixel_t pixel)
		{
			assert(x < mWidth && y < mHeight && z < mChannelSize);
			
			mImage[z][x][y] = pixel;
		}

		inline bool isSquare() const
		{
			return mWidth == mHeight;
		}

		inline void MaximizeImage(size_t width, size_t height)
		{
			if (mWidth != width && mHeight != height)
			{
				double rate = static_cast<double>(height) / static_cast<double>(mHeight);
				pixel_t*** newImage = new pixel_t**[mChannelSize];
				for (size_t z = 0; z < mChannelSize; ++z)
				{
					newImage[z] = new pixel_t*[width];
					for (size_t x = 0; x < width; ++x)
					{
						newImage[z][x] = new pixel_t[height];
						for (size_t y = 0; y < height; ++y)
						{
							size_t newX = static_cast<size_t>(static_cast<double>(x) / rate);
							size_t newY = static_cast<size_t>(static_cast<double>(y) / rate);

							double rightX = static_cast<double>(x) / rate - static_cast<double>(newX);
							double leftX = 1 - rightX;
							double bottomY = static_cast<double>(y) / rate - static_cast<double>(newY);
							double topY = 1 - bottomY;

							double topLeftWeight = leftX * topY;
							double topRightWeight = rightX * topY;
							double bottomRightWeight = rightX * bottomY;
							double bottomLeftWeight = leftX * bottomY;

							switch (((newX + 1 < mWidth) << 1) + (newY + 1 < mHeight))
							{
							case 0b10:
								newImage[z][x][y] = leftX * mImage[z][newX][newY] + rightX * mImage[z][newX + 1][newY];
								break;
							case 0b11:
								newImage[z][x][y] = topLeftWeight * mImage[z][newX][newY] + topRightWeight * mImage[z][newX + 1][newY] + bottomLeftWeight * mImage[z][newX][newY + 1] + bottomRightWeight * mImage[z][newX + 1][newY + 1];
								break;
							case 0b01:
								newImage[z][x][y] = topY * mImage[z][newX][newY] + bottomY * mImage[z][newX][newY + 1];
								break;
							case 0b00:
								newImage[z][x][y] = mImage[z][newX][newY];
								break;
							default:
								assert(false);
								break;
							}
						}
					}
				}

				for (size_t z = 0; z < mChannelSize; ++z)
				{
					for (size_t x = 0; x < mWidth; ++x)
					{
						delete[] mImage[z][x];
					}
					delete[] mImage[z];
				}
				delete[] mImage;

				mImage = newImage;
				mWidth = width;
				mHeight = height;
			}
		}

		inline void ResizeImage(size_t width, size_t height)
		{
			assert(width > 0 && height > 0);
			if (mImage != nullptr && (mWidth != width || mHeight != height))
			{
				for (size_t z = 0; z < mChannelSize; ++z)
				{
					if (mWidth != width)
					{
						for (size_t x = 0; x < mWidth; ++x)
						{
							delete[] mImage[z][x];
						}
						delete[] mImage[z];
						mImage[z] = new pixel_t*[width];
						mHeight = height;
						for (size_t x = 0; x < width; ++x)
						{
							mImage[z][x] = new pixel_t[mHeight];
							for (size_t y = 0; y < mHeight; ++y)
							{
								mImage[z][x][y] = 0.0;
							}
						}

						mWidth = width;
					}
					else
					{
						for (size_t x = 0; x < mWidth; ++x)
						{
							if (mHeight != height)
							{
								delete[] mImage[z][x];
							}

							mImage[z][x] = new pixel_t[height];
							for (size_t y = 0; y < height; ++y)
							{
								mImage[z][x][y] = 0.0;
							}
						}

						mHeight = height;
					}
				}
			}
		}

		inline uint8_t GetCoarseLabel() const
		{
			return mCoarseLabel;
		}

		inline uint8_t GetFineLabel() const
		{
			return mFineLabel;
		}

		inline void PrintValues(std::ostream& os) const
		{
			os << "PRINTING IMAGE: " << std::endl;
			for (size_t z = 0; z < mChannelSize; ++z)
			{
				os << "CHANNEL " << z << std::endl;
				for (size_t x = 0; x < mWidth; ++x)
				{
					for (size_t y = 0; y < mHeight; ++y)
					{
						os << std::setw(16) << mImage[x][y][z] << " ";
					}
					os << std::endl;
				}
				os << std::endl;
			}
		}
	private:
		size_t mChannelSize;
		size_t mWidth;
		size_t mHeight;
		pixel_t*** mImage;
		uint8_t mCoarseLabel;
		uint8_t mFineLabel;
	};
}