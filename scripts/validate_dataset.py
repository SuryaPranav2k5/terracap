"""Runner script to execute validation and save the final report."""

from pathlib import Path
from image_recognition.validator import DatasetValidator, format_markdown_report


def main():
    print("Starting full validation of NWPU-Caption dataset...")
    validator = DatasetValidator(dataset_dir="dataset")
    results = validator.run_all(compute_perceptual_hashes=True)

    report_content = format_markdown_report(results)

    # 1. Save to project workspace
    workspace_report_path = Path("reports/validation_report.md")
    workspace_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(workspace_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved workspace report to {workspace_report_path}")

    # 2. Save to artifact directory
    artifact_dir = Path(r"C:\Users\prana\.gemini\antigravity-ide\brain\bcf13d3e-1aa4-4555-b5b7-79242f95c72f")
    if artifact_dir.exists():
        artifact_report_path = artifact_dir / "dataset_validation_report.md"
        with open(artifact_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Saved artifact report to {artifact_report_path}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Disk Images: {results.total_disk_images:,}")
    print(f"Total Referenced Images: {results.total_json_entries:,}")
    print(f"Missing Images: {len(results.missing_on_disk)}")
    print(f"Corrupted Images: {len(results.corrupted_images)}")
    print(f"Exact Byte Image Duplicates: {len(results.exact_image_duplicate_groups)} groups")
    print(f"Perceptual (dHash) Image Duplicates: {len(results.perceptual_image_duplicate_pairs)} pairs")
    print(f"Image Split Leakage Instances: {len(results.image_split_leakage)}")
    print(f"Vocabulary Size: {results.vocab_size:,} unique words")
    detached_periods = results.punctuation_issues.get("detached_period_spaces (' .')", 0)
    print(f"Detached Period Formatting Artifacts (' .'): {detached_periods:,}")
    print(f"Cross-Image Duplicate Captions: {len(results.exact_caption_duplicate_groups):,}")
    print(f"Caption Split Leakage (Cross-Split Duplicate Phrases): {results.caption_split_leakage_count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
