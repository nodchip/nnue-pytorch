import visualize


def test_select_input_layout_for_chess_halfkav2_hm():
    assert visualize.select_input_layout("HalfKAv2_hm", 22528) == "chess"
    assert visualize.select_input_layout("HalfKAv2_hm^", 22528) == "chess"


def test_select_input_layout_for_shogi_halfka_hm():
    assert visualize.select_input_layout("HalfKA_hm", 73305) == "generic"
    assert visualize.select_input_layout("HalfKA_hm^", 73305) == "generic"
