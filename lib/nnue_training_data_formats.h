/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include "..\YaneuraOu\source\learn\learn.h"
#include "..\YaneuraOu\source\position.h"
#include "..\YaneuraOu\source\thread.h"
#include "..\YaneuraOu\source\types.h"

namespace shogi {
    struct TrainingDataEntry
    {
        std::shared_ptr<StateInfo> stateInfo = std::make_shared<StateInfo>();
        std::shared_ptr<Position> pos = std::make_shared<Position>();
        Move move;
        std::int16_t score;
        std::uint16_t ply;
        std::int16_t result;
    };

    // 自玉が敵陣に入ったときのボーナスを計算する。
    Value CalculateEnteringKingBonus(const Position& pos, Color color) {
        // 敵陣
        Bitboard ef = enemy_field(color);

        // (b) 自玉がそもそも敵陣にいなければ 0 を返す。
        if (!(ef & pos.king_square(color)))
            return VALUE_ZERO;

        // (d) 敵陣に存在する自分の駒の数（歩〜金など軽い駒）
        int p1 = (pos.pieces(color) & ef).pop_count();

        // 敵陣に存在する自分の角/馬/飛/龍などの駒の数
        int p2 = ((pos.pieces(color, BISHOP_HORSE, ROOK_DRAGON)) & ef).pop_count();

        // 自玉1点、軽い駒5点、重い駒は（角・飛）1枚につき 5 点加算して評価する。
        // 基本式： 自駒の数 + （重い駒の数 × 4） - 1

        // (c)
        // 手駒の点数を加算する。
        // ・歩の場合は 1 点
        // ・香・桂・銀・金も 1 点
        // ・角・飛はそれぞれ 5 点
        Hand h = pos.hand_of(color);
        int score = p1 + p2 * 4 - 1
            + hand_count(h, PAWN) + hand_count(h, LANCE)
            + hand_count(h, KNIGHT) + hand_count(h, SILVER)
            + hand_count(h, GOLD)
            + (hand_count(h, BISHOP) + hand_count(h, ROOK)) * 5;

        // 最終スコアを返す
        return static_cast<Value>(p1 + score);
    }

    constexpr const int EnteringKingBonusFactor = 20;

    [[nodiscard]] inline TrainingDataEntry packedSfenValueToTrainingDataEntry(const Learner::PackedSfenValue& psv)
    {
        TrainingDataEntry ret;

        ret.pos->set_from_packed_sfen(psv.sfen, &*ret.stateInfo, Threads.main());
        ret.move = ret.pos->to_move(Move16(psv.move));
        ret.score = psv.score;
        ret.ply = psv.gamePly;
        ret.result = psv.game_result;

        return ret;
    }
}
