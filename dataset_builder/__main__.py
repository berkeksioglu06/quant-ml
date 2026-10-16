import argparse
import json
import sys
from pathlib import Path

from .labels import Costs
from .pipeline import build


def main():
    parser = argparse.ArgumentParser(
        description="Quant ML özellik ve etiket üreticisi — yerel inceleme sürümü"
    )
    parser.add_argument(
        "--project", type=Path, required=True, help="Piyasa veri projesinin kök klasörü"
    )
    parser.add_argument("--snapshot", required=True, help="Örnek: yahoo_20261016T121656Z")
    parser.add_argument(
        "--output", type=Path, required=True, help="Henüz var olmayan ayrı çıktı klasörü"
    )
    parser.add_argument("--layout", choices=["auto", "legacy", "main"], default="auto")
    parser.add_argument("--as-of", help="İsteğe bağlı geçmiş kesim tarihi: YYYY-MM-DD")
    parser.add_argument("--buy-bps", type=float, help="Alış toplam maliyeti; 1 bps = %%0.01")
    parser.add_argument("--sell-bps", type=float, help="Satış toplam maliyeti; 1 bps = %%0.01")
    parser.add_argument(
        "--neutral-threshold", type=float, help="İsteğe bağlı net getiri eşiği; 0.01 = %%1"
    )
    args = parser.parse_args()
    if (args.buy_bps is None) != (args.sell_bps is None):
        parser.error("Alış ve satış maliyetlerini birlikte belirtin")
    try:
        costs = Costs(args.buy_bps, args.sell_bps) if args.buy_bps is not None else None
        result = build(
            args.project,
            args.snapshot,
            args.output,
            layout=args.layout,
            as_of=args.as_of,
            costs=costs,
            neutral_threshold=args.neutral_threshold,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Üretim tamamlanamadı: {exc}\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "snapshot", "layout", "rows", "symbols", "last_date")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"İnceleme dosyası: {args.output.resolve() / 'INCELEME.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
