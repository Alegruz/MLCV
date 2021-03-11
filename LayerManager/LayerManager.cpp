#include "LayerManager.h"

namespace mlcv
{
	LayerManager::LayerManager()
		: mTopLayer(nullptr)
	{
	}

	void LayerManager::AddLayer(Layer*&& layer)
	{
		Layer* iterateLayer = mTopLayer;
		while (iterateLayer != nullptr)
		{
			if (iterateLayer->mNextLayer != nullptr)
			{
				iterateLayer = iterateLayer->mNextLayer;
			}
			else
			{
				break;
			}
		}

		iterateLayer->mNextLayer = layer;
		iterateLayer->mNextLayer->mPrevLayer = iterateLayer;
		layer = nullptr;
	}

	void LayerManager::ProcessLayers()
	{
		Layer* iterateLayer = mTopLayer;
		while (iterateLayer != nullptr)
		{
			iterateLayer->Process();
			iterateLayer->Activate();
			iterateLayer->sendDataToNextLayer();
			iterateLayer = iterateLayer->mNextLayer;
		}
	}
}