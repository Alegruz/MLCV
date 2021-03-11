#pragma once

#include "Defs.h"
#include "Layer.h"

namespace mlcv
{
	struct LayerManager
	{
	public:
		inline LayerManager();
		virtual void AddLayer(Layer*&& layer);
		virtual void ProcessLayers();
	private:
		Layer* mTopLayer;
	};
}