import argparse
import json

from dataset_builder.verify import verify_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Üretilmiş dosyaların bütünlüğünü ve veri sınırlarını denetle"
    )
    parser.add_argument("directory")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_output(args.directory), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Doğrulama başarısız: {exc}\n")
