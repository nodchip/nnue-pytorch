from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import font_manager

PIECE_TO_KANJI = {
    "P": "歩",
    "L": "香",
    "N": "桂",
    "S": "銀",
    "G": "金",
    "B": "角",
    "R": "飛",
    "K": "玉",
    "+P": "と",
    "+L": "杏",
    "+N": "圭",
    "+S": "全",
    "+B": "馬",
    "+R": "龍",
}

HAND_ORDER = ["R", "B", "G", "S", "N", "L", "P"]

_FONT_READY = False
JAPANESE_FONT_CANDIDATES = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAGothic",
]


def progress_to_bucket(progress: float, bucket_count: int = 8) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    idx = int(progress * bucket_count)
    if idx < 0:
        return 0
    if idx >= bucket_count:
        return bucket_count - 1
    return idx


def piece_to_kanji(piece: str) -> str:
    normalized = piece.upper()
    if normalized not in PIECE_TO_KANJI:
        raise ValueError(f"unsupported piece token: {piece}")
    return PIECE_TO_KANJI[normalized]


def parse_sfen_hands(hand_field: str) -> tuple[dict[str, int], dict[str, int]]:
    sente = defaultdict(int)
    gote = defaultdict(int)
    if hand_field == "-":
        return dict(sente), dict(gote)

    number = ""
    for ch in hand_field:
        if ch.isdigit():
            number += ch
            continue

        count = int(number) if number else 1
        number = ""
        piece = ch.upper()
        if ch.isupper():
            sente[piece] += count
        else:
            gote[piece] += count
    return dict(sente), dict(gote)


def parse_sfen_board(sfen: str) -> list[list[str]]:
    parts = sfen.strip().split()
    if len(parts) < 1:
        raise ValueError("invalid sfen")
    rows = parts[0].split("/")
    if len(rows) != 9:
        raise ValueError("invalid sfen board rows")

    board: list[list[str]] = []
    for row in rows:
        parsed: list[str] = []
        i = 0
        while i < len(row):
            c = row[i]
            if c.isdigit():
                parsed.extend([""] * int(c))
                i += 1
                continue
            if c == "+":
                if i + 1 >= len(row):
                    raise ValueError("invalid promoted piece in sfen")
                parsed.append("+" + row[i + 1])
                i += 2
                continue
            parsed.append(c)
            i += 1
        if len(parsed) != 9:
            raise ValueError("invalid sfen row width")
        board.append(parsed)
    return board


def parse_sfen_position(sfen: str) -> tuple[list[list[str]], dict[str, int], dict[str, int]]:
    parts = sfen.strip().split()
    if len(parts) < 4:
        raise ValueError("invalid sfen")
    board = parse_sfen_board(sfen)
    sente_hands, gote_hands = parse_sfen_hands(parts[2])
    return board, sente_hands, gote_hands


def format_hands_text(label: str, hands: dict[str, int]) -> str:
    tokens = []
    for p in HAND_ORDER:
        count = hands.get(p, 0)
        if count <= 0:
            continue
        piece = piece_to_kanji(p)
        tokens.append(f"{piece}{count if count > 1 else ''}")
    body = " ".join(tokens) if tokens else "なし"
    return f"{label}: {body}"


def choose_japanese_font(available_fonts: set[str] | None = None) -> str:
    if available_fonts is None:
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in JAPANESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            return font_name
    return "DejaVu Sans"


def _configure_japanese_font() -> None:
    global _FONT_READY
    if _FONT_READY:
        return
    plt.rcParams["font.family"] = [choose_japanese_font()]
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def draw_sfen_on_axis(ax, sfen: str, title: str) -> None:
    _configure_japanese_font()
    board, sente_hands, gote_hands = parse_sfen_position(sfen)
    ax.set_xlim(0, 9)
    ax.set_ylim(-1.2, 10.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f2dfb0")

    for y in range(10):
        ax.plot([0, 9], [y, y], color="black", linewidth=0.8)
        ax.plot([y, y], [0, 9], color="black", linewidth=0.8)

    for r in range(9):
        for c in range(9):
            piece = board[r][c]
            if not piece:
                continue
            text_color = "#111111" if piece[-1].isupper() else "#1f4f8a"
            rotation = 0 if piece[-1].isupper() else 180
            ax.text(
                c + 0.5,
                8.5 - r,
                piece_to_kanji(piece),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                rotation=rotation,
            )

    ax.text(4.5, 9.45, format_hands_text("後手", gote_hands), ha="center", va="center", fontsize=8)
    ax.text(4.5, -0.45, format_hands_text("先手", sente_hands), ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=8)


def make_grid_figure(
    items: list[tuple[str, float]],
    output_path: str,
    cols: int = 5,
    title: str | None = None,
) -> None:
    if cols <= 0:
        raise ValueError("cols must be positive")
    if not items:
        fig = plt.figure(figsize=(6, 3))
        fig.suptitle(title or "No positions")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    elif cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_list):
        if i >= len(items):
            ax.axis("off")
            continue
        sfen, progress = items[i]
        draw_sfen_on_axis(ax, sfen=sfen, title=f"p={progress:.3f}")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
