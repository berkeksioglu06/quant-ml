import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .data import load_panel
from .experiment import run

DEFAULT_DATASET = Path("output/yahoo_20261016T121656Z_dataset_v1_1")


def main():
    parser = argparse.ArgumentParser(
        description="Aşama 1 · Deney 1: kıyaslar + Ridge, K1–K4 (test dönemi kapalı)"
    )
    parser.add_argument("--project", type=Path, default=Path("."), help="Repo kök klasörü")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Hazır veri seti")
    parser.add_argument("--horizon", type=int, default=20, help="Etiket ufku (seans)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Henüz var olmayan çıktı klasörü; varsayılan output/experiments/",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    dataset = args.dataset if args.dataset.is_absolute() else project / args.dataset
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or project / "output/experiments" / f"{stamp}_ridge_h{args.horizon}"
    try:
        panel = load_panel(dataset, project, args.horizon)
        config, summary, _ = run(panel, output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Deney tamamlanamadı: {exc}\n")
    ridge = summary.loc[summary["model"].eq("ridge")].set_index("target")
    print(
        json.dumps(
            {
                "run_id": config["run_id"],
                "horizon": config["horizon"],
                "ridge_rank_ic": ridge["ic_mean"].round(4).to_dict(),
                "ridge_r2_oos": ridge["r2_oos"].round(4).to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Rapor: {Path(output).resolve() / 'RAPOR.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
