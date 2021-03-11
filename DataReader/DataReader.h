#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "Defs.h"

#include "Image.h"

namespace mlcv
{
	struct DataReader
	{
	public:
		DataReader(const char* path);
		virtual void PrintDataSet(std::ostream& os, bool isPrintEnabled = false) const = 0;
		virtual Image* GetSingleImageMalloc(size_t index, bool isPrintEnabled = false) = 0;
	protected:
		const char* mPath;
	};

	struct CifarReader : public DataReader
	{
	public:
		enum class eCifarSuperclass
		{
			AQUATIC_MAMMALS,
			FISH,
			FLOWERS,
			FOOD_CONTAINERS,
			FRUIT_AND_VEGETABLES,
			HOUSEHOLD_ELECTRICAL_DEVICES,
			HOUSEHOLD_FURNITURE,
			INSECTS,
			LARGE_CARNIVORES,
			LARGE_MAN_MADE_OUTDOOR_THINGS,
			LARGE_NATURAL_OUTDOOR_SCENES,
			LARGE_OMNIVORES_AND_HERBIVORES,
			MEDIUM_MAMMALS,
			NON_INSECT_INVERTEBRATES,
			PEOPLE,
			REPTILES,
			SMALL_MAMMALS,
			TREES,
			VEHICLES_1,
			VEHICLES_2
		};

		enum class eCifarClass
		{
			APPLE,
			AQUARIUM_FISH,
			BABY,
			BEAR,
			BEAVER,
			BED,
			BEE,
			BEETLE,
			BICYCLE,
			BOTTLE,
			BOWL,
			BOY,
			BRIDGE,
			BUS,
			BUTTERFLY,
			CAMEL,
			CAN,
			CASTLE,
			CATERPILLAR,
			CATTLE,
			CHAIR,
			CHIMPANZEE,
			CLOCK,
			CLOUD,
			COCKROACH,
			COUCH,
			CRAB,
			CROCODILE,
			CUP,
			DINOSAUR,
			DOLPHIN,
			ELEPHANT,
			FLATFISH,
			FOREST,
			FOX,
			GIRL,
			HAMSTER,
			HOUSE,
			KANGAROO,
			KEYBOARD,
			LAMP,
			LAWN_MOWER,
			LEOPARD,
			LION,
			LIZARD,
			LOBSTER,
			MAN,
			MAPLE_TREE,
			MOTORCYCLE,
			MOUNTAIN,
			MOUSE,
			MUSHROOM,
			OAK_TREE,
			ORANGE,
			ORCHID,
			OTTER,
			PALM_TREE,
			PEAR,
			PICKUP_TRUCK,
			PINE_TREE,
			PLAIN,
			PLATE,
			POPPY,
			PORCUPINE,
			POSSUM,
			RABBIT,
			RACCOON,
			RAY,
			ROAD,
			ROCKET,
			ROSE,
			SEA,
			SEAL,
			SHARK,
			SHREW,
			SKUNK,
			SKYSCRAPER,
			SNAIL,
			SNAKE,
			SPIDER,
			SQUIRREL,
			STREETCAR,
			SUNFLOWER,
			SWEET_PEPPER,
			TABLE,
			TANK,
			TELEPHONE,
			TELEVISION,
			TIGER,
			TRACTOR,
			TRAIN,
			TROUT,
			TULIP,
			TURTLE,
			WARDROBE,
			WHALE,
			WILLOW_TREE,
			WOLF,
			WOMAN,
			WORM
		};

		CifarReader(const char* path);
		virtual void PrintDataSet(std::ostream& os, bool isPrintEnabled = false) const;
		virtual Image* GetSingleImageMalloc(size_t index, bool isPrintEnabled = false);
	};
}