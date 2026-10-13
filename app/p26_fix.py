# -*- coding: utf-8 -*-
"""P26 手順 2: 歯止めの 2 段目（`p26_arguments`）を、これまでの行で確かめる。学習なし・端末なし。**どの行も直す用**（arg_eval_v3 も、P25 の結果を見たので直す用）。

設計: DoC/設計書_P26_歯止めの2段目.md の 4 節。中身は `p25_fix` と同じ（比べる引数を P23・P25・P26 にし、保存先を results/P26 にしただけ）。

    python -m app.p26_fix        # results/P26/fix.md・fix.json
"""
from __future__ import annotations

from . import p25_fix as base
from .common import PROJECT_ROOT
from .p23_arguments import make_resolve as args_p23
from .p25_arguments import make_resolve as args_p25
from .p26_arguments import make_resolve as args_p26


def main() -> None:
    base.RES = PROJECT_ROOT / "results" / "P26"
    # 1 番目と最後を比べる（止めた行は P25 → P26 で見たいので、P25 を 1 番目に置く）
    base.CONFIGS = [("P25（歯止め 1 段目）", args_p25), ("P26（歯止め 2 段目）", args_p26)]
    base.main()


if __name__ == "__main__":
    main()
