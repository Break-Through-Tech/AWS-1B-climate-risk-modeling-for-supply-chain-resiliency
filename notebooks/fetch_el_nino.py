from pathlib import Path
from ucimlrepo import fetch_ucirepo

el_nino = fetch_ucirepo(id=122)

X = el_nino.data.features
y = el_nino.data.targets

output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(exist_ok=True)

X.to_csv(output_dir / "el_nino_features.csv", index=False)

if not y.empty:
    y.to_csv(output_dir / "el_nino_targets.csv", index=False)

print(f"Saved CSV files to {output_dir}")