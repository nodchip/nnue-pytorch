#ifndef _TANUKI_PROGRESS_H
#define _TANUKI_PROGRESS_H

#include <map>
#include <string>

#include "YaneuraOu/source/evaluate.h"
#include "YaneuraOu/source/usi.h"

class Position;

namespace Tanuki {
	class Progress {
	public:
		static bool Initialize(USI::OptionsMap& o);
		bool Load();
		bool Save();
		bool Learn();
		double Estimate(const Position& pos);

	private:
		double weights_[SQ_NB][Eval::BonaPiece::fe_end] = {};
	};
}

#endif
