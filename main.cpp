#include <string.h>

#include <iostream>
#include <string>

#include "MlcvConfig.h"

#include "lodepng.h"

#include "Defs.h"
#include "DataReader.h"
#include "Layer.h"

void vggD(mlcv::Image*&& image);

int main(int argc, char** argv)
{
	std::cout << MLCV_VERSION_MAJOR << std::endl;

	mlcv::CifarReader reader("cifar-100-binary/train.bin");
	mlcv::Image* image = reader.GetSingleImageMalloc(0, true);
	image->MaximizeImage(224, 224);

	char fileName[15];
	snprintf(fileName, 15, "resized.png");
	uint8_t* eightBitImage = image->GetLodePngCompatibleImageMalloc();
	unsigned int error = lodepng::encode(fileName, eightBitImage, 224, 224, LodePNGColorType::LCT_RGB);

	delete[] eightBitImage;

	for (size_t i = 0; i < 1; ++i)
	{
		vggD(std::move(image), 0.1);
	}

	// CIFAR100 32x32 -> VGG 224*224
	// Cost Function
	// Back Propagation

	return 0;
}

void vggD(mlcv::Image*&& image, double learningRate)
{
	uint8_t answer = image->GetFineLabel();
	// 1st floor
	std::cout << "1ST FLOOR" << std::endl;
	size_t outputSize = 64;
	mlcv::Filter** filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	mlcv::ActivationFunction* reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer1(outputSize, std::move(filters), std::move(image), std::move(reLu));
	convLayer1.Process();
	convLayer1.Activate();

	// 2nd floor
	std::cout << "2ND FLOOR" << std::endl;
	outputSize = 64;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer2(outputSize, std::move(filters), convLayer1, std::move(reLu));
	convLayer2.Process();
	convLayer2.Activate();

	mlcv::MaxPoolLayer maxPoolLayer2(2, 2, convLayer2);
	maxPoolLayer2.Process();

	// 3rd floor
	std::cout << "3RD FLOOR" << std::endl;
	outputSize = 128;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer3(outputSize, std::move(filters), maxPoolLayer2, std::move(reLu));
	convLayer3.Process();
	convLayer3.Activate();

	// 4th floor
	std::cout << "4TH FLOOR" << std::endl;
	outputSize = 128;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer4(outputSize, std::move(filters), convLayer3, std::move(reLu));
	convLayer4.Process();
	convLayer4.Activate();

	mlcv::MaxPoolLayer maxPoolLayer4(2, 2, convLayer4);
	maxPoolLayer4.Process();

	// 5th Floor
	std::cout << "5TH FLOOR" << std::endl;
	outputSize = 256;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer5(outputSize, std::move(filters), maxPoolLayer4, std::move(reLu));
	convLayer5.Process();
	convLayer5.Activate();

	// 6th Floor
	std::cout << "6TH FLOOR" << std::endl;
	outputSize = 256;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer6(outputSize, std::move(filters), convLayer5, std::move(reLu));
	convLayer6.Process();
	convLayer6.Activate();

	// 7th Floor
	std::cout << "7TH FLOOR" << std::endl;
	outputSize = 256;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer7(outputSize, std::move(filters), convLayer6, std::move(reLu));
	convLayer7.Process();
	convLayer7.Activate();

	mlcv::MaxPoolLayer maxPoolLayer7(2, 2, convLayer7);
	maxPoolLayer7.Process();

	// 8th Floor
	std::cout << "8TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer8(outputSize, std::move(filters), maxPoolLayer7, std::move(reLu));
	convLayer8.Process();
	convLayer8.Activate();

	// 9th Floor
	std::cout << "9TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer9(outputSize, std::move(filters), convLayer8, std::move(reLu));
	convLayer9.Process();
	convLayer9.Activate();

	// 10th Floor
	std::cout << "10TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer10(outputSize, std::move(filters), convLayer9, std::move(reLu));
	convLayer10.Process();
	convLayer10.Activate();

	mlcv::MaxPoolLayer maxPoolLayer10(2, 2, convLayer10);
	maxPoolLayer10.Process();

	// 11th Floor
	std::cout << "11TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer11(outputSize, std::move(filters), maxPoolLayer10, std::move(reLu));
	convLayer11.Process();
	convLayer11.Activate();

	// 12th Floor
	std::cout << "12TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer12(outputSize, std::move(filters), convLayer11, std::move(reLu));
	convLayer12.Process();
	convLayer12.Activate();

	// 13th Floor
	std::cout << "13TH FLOOR" << std::endl;
	outputSize = 512;
	filters = new mlcv::Filter*[outputSize];
	for (size_t i = 0; i < outputSize; ++i)
	{
		filters[i] = new mlcv::Filter(3, 3, outputSize, 0.0);
		filters[i]->RandomizeByNormalDistribution(0.0, 0.01);
	}
	reLu = new mlcv::ReLu();
	mlcv::ConvLayer convLayer13(outputSize, std::move(filters), convLayer12, std::move(reLu));
	convLayer13.Process();
	convLayer13.Activate();

	mlcv::MaxPoolLayer maxPoolLayer13(2, 2, convLayer13);
	maxPoolLayer13.Process();

	// 14th Floor
	std::cout << "14TH FLOOR" << std::endl;
	double mean = 0.0;
	double standardDeviation = 0.01;
	std::random_device randomDevice;
	std::default_random_engine generator(randomDevice());
	std::normal_distribution<double> distribution(mean, standardDeviation);
	size_t inputSize = 25088;
	outputSize = 4096;
	double** weights = new double*[inputSize];
	for (size_t i = 0; i < inputSize; ++i)
	{
		weights[i] = new double[outputSize];
		for (size_t j = 0; j < outputSize; ++j)
		{
			weights[i][j] = distribution(generator);
		}
	}
	reLu = new mlcv::ReLu();
	mlcv::FullyConnectedLayer fcLayer14(inputSize, outputSize, maxPoolLayer13, std::move(weights), nullptr, std::move(reLu));
	fcLayer14.Process();
	fcLayer14.Activate();

	// 15th Floor
	std::cout << "15TH FLOOR" << std::endl;
	inputSize = 4096;
	outputSize = 4096;
	weights = new double*[inputSize];
	for (size_t i = 0; i < inputSize; ++i)
	{
		weights[i] = new double[outputSize];
		for (size_t j = 0; j < outputSize; ++j)
		{
			weights[i][j] = distribution(generator);
		}
	}
	reLu = new mlcv::ReLu();
	mlcv::FullyConnectedLayer fcLayer15(inputSize, outputSize, fcLayer14, std::move(weights), nullptr, std::move(reLu), true);
	fcLayer15.Process();
	fcLayer15.Activate();

	// 16th Floor
	std::cout << "16TH FLOOR" << std::endl;
	inputSize = 4096;
	outputSize = 100;
	weights = new double*[inputSize];
	for (size_t i = 0; i < inputSize; ++i)
	{
		weights[i] = new double[outputSize];
		for (size_t j = 0; j < outputSize; ++j)
		{
			weights[i][j] = distribution(generator);
		}
	}
	mlcv::SoftMax* softMax = new mlcv::SoftMax();
	mlcv::FullyConnectedLayer fcLayer16(inputSize, outputSize, fcLayer15, std::move(weights), nullptr, std::move(softMax), false);
	fcLayer16.Process();
	fcLayer16.Activate();

	std::ofstream resultFile("result.txt");

	double max = fcLayer16[0];
	size_t i = 0;
	for (; i < 100; ++i)
	{
		if (max <= fcLayer16[i])
		{
			max = fcLayer16[i];
		}
		resultFile << "[" << i << "] = " << fcLayer16[i] << std::endl;
	}
	resultFile << "MAX[" << i << "]: " << max << " ~ " << maxPoolLayer13.GetCoarseLabel() << " : " << maxPoolLayer13.GetFineLabel() << std::endl;

	if (i != static_cast<size_t>(answer))
	{

	}

	resultFile.close();
}