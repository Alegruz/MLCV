#pragma once

#include <random>

#include "Defs.h"

namespace mlcv
{
	struct Filter
	{
	public:
		inline explicit Filter(size_t width, size_t height, pixel_t initialValue)
			: Filter(width, height, 1, initialValue)
		{
		}

		inline explicit Filter(size_t width, size_t height, size_t length, pixel_t initialValue)
			: mWidth(width)
			, mHeight(height)
			, mLength(length)
		{
			mFilter = new pixel_t**[mWidth];
			for (size_t x = 0; x < mWidth; ++x)
			{
				mFilter[x] = new pixel_t*[mHeight];
				for (size_t y = 0; y < mHeight; ++y)
				{
					mFilter[x][y] = new pixel_t[mLength];
					for (size_t z = 0; z < mLength; ++z)
					{
						mFilter[x][y][z] = initialValue;
					}
				}
			}
		}

		inline explicit Filter(const Filter& other)
		{
			if (&other != this)
			{
				mFilter = new pixel_t**[mWidth];
				for (size_t x = 0; x < mWidth; ++x)
				{
					mFilter[x] = new pixel_t*[mHeight];
					for (size_t y = 0; y < mHeight; ++y)
					{
						mFilter[x][y] = new pixel_t[mLength];
						for (size_t z = 0; z < mLength; ++z)
						{
							mFilter[x][y][z] = other.mFilter[x][y][z];
						}
					}
				}
			}
		}

		virtual inline Filter& operator=(const Filter& other)
		{
			if (&other != this)
			{
				if (mFilter != nullptr)
				{
					if (mHeight != other.mHeight)
					{
						for (size_t x = 0; x < mHeight; ++x)
						{
							delete[] mFilter[x];
						}

						if (mWidth != other.mWidth)
						{
							delete[] mFilter;
						}

						mFilter = nullptr;
					}
				}

				mWidth = other.mWidth;
				mHeight = other.mHeight;
				mLength = other.mLength;

				if (mFilter == nullptr)
				{
					mFilter = new pixel_t**[mWidth];
					for (size_t x = 0; x < mWidth; ++x)
					{
						mFilter[x] = new pixel_t*[mHeight];
						for (size_t y = 0; y < mHeight; ++y)
						{
							mFilter[x][y] = new pixel_t[mLength];
							for (size_t z = 0; z < mLength; ++z)
							{
								mFilter[x][y][z] = other.mFilter[x][y][z];
							}
						}
					}
				}
				else
				{
					for (size_t x = 0; x < mWidth; ++x)
					{
						for (size_t y = 0; y < mHeight; ++y)
						{
							for (size_t z = 0; z < mLength; ++z)
							{
								mFilter[x][y][z] = other.mFilter[x][y][z];
							}
						}
					}
				}
			}

			return *this;
		}

		virtual inline ~Filter()
		{
			if (mFilter != nullptr)
			{
				for (size_t x = 0; x < mWidth; ++x)
				{
					if (mFilter[x] != nullptr)
					{
						for (size_t y = 0; y < mHeight; ++y)
						{
							delete[] mFilter[x][y];
						}

						delete[] mFilter[x];
					}
				}

				delete[] mFilter;
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

		inline size_t GetLength() const
		{
			return mLength;
		}

		inline virtual bool isSquare() const
		{
			return mWidth == mHeight;
		}

		void SetValue(size_t x, size_t y, pixel_t value)
		{
			SetValue(x, y, 1, value);
		}

		void SetValue(size_t x, size_t y, size_t z, pixel_t value)
		{
			assert(x < mWidth && y < mHeight && z < mLength);

			mFilter[x][y][z] = value;
		}

		pixel_t GetValue(size_t x, size_t y) const
		{
			return GetValue(x, y, 1);
		}

		pixel_t GetValue(size_t x, size_t y, size_t z) const
		{
			assert(x < mWidth && y < mHeight && z < mLength);

			return mFilter[x][y][z];
		}

		void RandomizeByNormalDistribution(pixel_t mean, pixel_t standardDeviation)
		{
			std::random_device randomDevice;
			std::default_random_engine generator(randomDevice());
			std::normal_distribution<pixel_t> distribution(mean, standardDeviation);

			for (size_t x = 0; x < mWidth; ++x)
			{
				for (size_t y = 0; y < mHeight; ++y)
				{
					for (size_t z = 0; z < mLength; ++z)
					{
						mFilter[x][y][z] = distribution(generator);
					}
				}
			}
		}

		inline void PrintValues(std::ostream& os) const
		{
			os << "PRINTING FILTER: " << std::endl;
			for(size_t z = 0; z < mLength; ++z)
			{
				std::cout << "CHANNEL: " << z << std::endl;
				for (size_t x = 0; x < mWidth; ++x)
				{
					for (size_t y = 0; y < mHeight; ++y)
					{
						os << std::setw(16) << mFilter[x][y][z] << " ";
					}
				}
				os << std::endl;
			}
		}
	protected:
		size_t mWidth;
		size_t mHeight;
		size_t mLength;
		pixel_t*** mFilter;
	};

	struct SquareFilter : public Filter
	{
	public:
		inline explicit SquareFilter(size_t size, pixel_t initialValue)
			: Filter(size, size, size, initialValue)
			, mSize(size)
		{
		}

		inline explicit SquareFilter(const SquareFilter& other) = default;
		inline SquareFilter& operator=(const SquareFilter& other) = default;

		inline ~SquareFilter() = default;

		inline size_t GetSize() const
		{
			return mSize;
		}

		inline virtual bool isSquare() const
		{
			return true;
		}
	protected:
		size_t mSize;
	};
}