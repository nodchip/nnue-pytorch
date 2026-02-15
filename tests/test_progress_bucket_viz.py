import pytest

from progress_bucket_viz import (
    choose_japanese_font,
    parse_sfen_board,
    parse_sfen_position,
    piece_to_kanji,
    progress_to_bucket,
)


def test_parse_sfen_board_shape_and_pieces():
    sfen = "lnsgkgsnl/1r5b1/p1pppp1pp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    board = parse_sfen_board(sfen)
    assert len(board) == 9
    assert all(len(row) == 9 for row in board)
    assert board[0][0] == "l"
    assert board[1][1] == "r"
    assert board[7][1] == "B"
    assert board[8][8] == "L"


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, 0),
        (0.124, 0),
        (0.125, 1),
        (0.999, 7),
        (1.0, 7),
        (-0.5, 0),
        (2.0, 7),
    ],
)
def test_progress_to_bucket(progress, expected):
    assert progress_to_bucket(progress) == expected


def test_piece_to_kanji_promoted():
    assert piece_to_kanji("P") == "歩"
    assert piece_to_kanji("+P") == "と"
    assert piece_to_kanji("r") == "飛"
    assert piece_to_kanji("+b") == "馬"


def test_parse_sfen_position_includes_hands():
    sfen = "9/9/9/9/9/9/9/9/9 b 2R2Pbgsn3p 1"
    board, sente_hands, gote_hands = parse_sfen_position(sfen)
    assert len(board) == 9
    assert sente_hands["R"] == 2
    assert sente_hands["P"] == 2
    assert gote_hands["B"] == 1
    assert gote_hands["G"] == 1
    assert gote_hands["S"] == 1
    assert gote_hands["N"] == 1
    assert gote_hands["P"] == 3


def test_choose_japanese_font_prefers_available_japanese_font():
    selected = choose_japanese_font({"Arial", "Meiryo", "Times New Roman"})
    assert selected == "Meiryo"


def test_choose_japanese_font_falls_back_to_dejavu():
    selected = choose_japanese_font({"Arial", "Times New Roman"})
    assert selected == "DejaVu Sans"
