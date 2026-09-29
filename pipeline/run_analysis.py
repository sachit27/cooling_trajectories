"""Run the complete Zurich urban-heat analysis in dependency order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "v3_00_validate_inputs.py",
    "v3_01_night_metrics.py",
    "v3_02_synoptic.py",
    "v3_03_development.py",
    "v3_04_holdout.py",
    "v3_05_specification_curve.py",
    "v3_10_era5_validation.py",
    "v3_11_observable_stability.py",
    "v3_12_mechanism_deepdive.py",
    "v3_14_trajectory_framework.py",
    "v3_15_station_traits.py",
    "v3_16_fitnah_and_relaxation.py",
    "v3_17_metrology_temporal.py",
    "v3_18_spatial_rebuild.py",
    "v3_19_tmin_coverage_sensitivity.py",
    "v3_20_submission_corrections.py",
    "v3_23_temporal_sensitivity.py",
    "v3_06_priority_screen.py",
    "v3_25_revision_temporal.py",
    "v3_26_revision_spatial.py",
    "v3_08_verify_numbers.py",
    "v3_28_revision_figures.py",
]


def main() -> None:
    for directory in ("outputs_v3", "outputs_robust", "outputs_revision", "figures"):
        (ROOT / directory).mkdir(exist_ok=True)

    for stage in STAGES:
        script = ROOT / "pipeline" / stage
        print(f"\n>>> {stage}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    print("\nAnalysis complete. Results and figures are in the local output folders.")


if __name__ == "__main__":
    main()
