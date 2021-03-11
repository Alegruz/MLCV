#include "DataReader.h"

#include <cassert>

#include "lodepng.h"

namespace mlcv
{
	DataReader::DataReader(const char* path)
		: mPath(path)
	{
	}

	CifarReader::CifarReader(const char* path)
		: DataReader(path)
	{
	}

	void CifarReader::PrintDataSet(std::ostream& os, bool isPrintEnabled) const
	{
		std::ifstream dataFile(mPath, std::ios::in | std::ios::binary);
	
		uint8_t label[2] = { 0, };
		uint8_t pixelByUint8[CIFAR100_PIXEL_SIZE] = { 0, };
		size_t count = 0;

		while (!dataFile.eof())
		{
			dataFile.read(reinterpret_cast<char*>(label), 2);
			dataFile.read(reinterpret_cast<char*>(pixelByUint8), CIFAR100_PIXEL_SIZE);

			if (isPrintEnabled)
			{
				ConvertToLodePngCompatibleArray(pixelByUint8, CIFAR100_PIXEL_SIZE, eImageOrientation::STRAIGHT);

				char fileName[15];
				snprintf(fileName, 15, "out_%ld.png", count);
				unsigned int error = lodepng::encode(fileName, pixelByUint8, 32, 32, LodePNGColorType::LCT_RGB);
			}
			++count;
		}

		dataFile.close();

		os << "count: " << count;
	}

	Image* CifarReader::GetSingleImageMalloc(size_t index, bool isPrintEnabled)
	{
		std::ifstream dataFile(mPath, std::ios::in | std::ios::binary);
	
		uint8_t label[2] = { 0, };
		uint8_t pixelByUint8[CIFAR100_PIXEL_SIZE] = { 0, };
		pixel_t* pixelByDouble = nullptr;
		size_t count = 0;

		while (!dataFile.eof())
		{
			if (count != index)
			{
				dataFile.ignore(CIFAR100_PIXEL_SIZE + 2);
			}
			else
			{
				eImageOrientation imageOrientation = eImageOrientation::STRAIGHT;
				dataFile.read(reinterpret_cast<char*>(label), 2);
				dataFile.read(reinterpret_cast<char*>(pixelByUint8), CIFAR100_PIXEL_SIZE);

				if (isPrintEnabled)
				{
					pixelByDouble = ConvertToLodePngCompatibleArrayMalloc(pixelByUint8, CIFAR100_PIXEL_SIZE, eImageOrientation::STRAIGHT);

					char fileName[15];
					snprintf(fileName, 15, "out_%ld.png", count);
					unsigned int error = lodepng::encode(fileName, pixelByUint8, 32, 32, LodePNGColorType::LCT_RGB);
					imageOrientation = eImageOrientation::RGB;
				}

				return new Image(pixelByDouble, CIFAR100_IMAGE_SIZE, CIFAR100_IMAGE_SIZE, imageOrientation);
			}
			++count;
		}

		dataFile.close();

		return nullptr;
	}
}