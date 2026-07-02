import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results across professions")

    parser.add_argument(
        "--attribute_to_eval",
        type=str,
        default="gender",
        choices=["gender", "race", "age"],
        help="Bias attribute to aggregate: gender, race, or age"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Parent results directory containing profession subdirectories. "
             "e.g. './results' — will scan for spatm_gender_* or baseline_gender_* folders"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="spatm",
        choices=["spatm", "baseline"],
        help="Which results to aggregate: spatm or baseline"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    return parser.parse_args()


def parse_eval_file(eval_file):
    """
    Parse evaluation_{attribute}.txt produced by evaluate_clip.py
    Expected last 2 lines:
        For Class [...], KL Divergence is 0.020136
        CLIP Score is 17.965625
    Returns (kl, clip_score) or None if file is missing/malformed.
    """
    if not os.path.exists(eval_file):
        print(f"  [WARNING] Eval file not found: {eval_file}")
        return None

    with open(eval_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) < 2:
        print(f"  [WARNING] Eval file too short: {eval_file}")
        return None

    try:
        # Last line: "CLIP Score is 17.965625"
        clip_line = lines[-1]
        clip_score = float(clip_line.split()[-1])

        # Second-to-last line: "For Class [...], KL Divergence is 0.020136"
        kl_line = lines[-2]
        kl = float(kl_line.split()[-1])

        return kl, clip_score

    except (ValueError, IndexError) as e:
        print(f"  [WARNING] Could not parse eval file {eval_file}: {e}")
        print(f"  Last lines were: {lines[-3:]}")
        return None


if __name__ == '__main__':

    args = parse_args()

    EVAL_BIAS = args.attribute_to_eval
    ROOT = args.root_dir
    MODE = args.mode

    # =====================================================
    # Find all matching profession subdirectories
    # e.g. spatm_gender_doctor, baseline_gender_nurse, etc.
    # =====================================================
    prefix = f"{MODE}_{EVAL_BIAS}_"
    subdirs = sorted([
        d for d in os.listdir(ROOT)
        if d.startswith(prefix) and os.path.isdir(os.path.join(ROOT, d))
    ])

    if not subdirs:
        print(f"No directories found matching prefix '{prefix}' in {ROOT}")
        print(f"Available directories: {os.listdir(ROOT)}")
        exit(1)

    print(f"\n{'='*50}")
    print(f"Aggregating {MODE.upper()} results for attribute: {EVAL_BIAS}")
    print(f"Found {len(subdirs)} profession directories")
    print(f"{'='*50}\n")

    # =====================================================
    # Output file
    # =====================================================
    out_file = os.path.join(ROOT, f'{MODE}_{EVAL_BIAS}_summary.txt')

    results = []

    with open(out_file, 'w') as fout:

        fout.write(f"SPATM Evaluation Summary\n")
        fout.write(f"Mode      : {MODE}\n")
        fout.write(f"Attribute : {EVAL_BIAS}\n")
        fout.write(f"{'='*50}\n\n")
        fout.write(f"{'Profession':<25} {'KL Div':>10} {'CLIP':>10}\n")
        fout.write(f"{'-'*50}\n")

        for subdir in subdirs:
            profession = subdir.replace(prefix, '').replace('_', ' ')
            eval_file = os.path.join(ROOT, subdir, f'evaluation_{EVAL_BIAS}.txt')

            parsed = parse_eval_file(eval_file)

            if parsed is None:
                print(f"  Skipping {subdir} — could not parse eval file")
                fout.write(f"{profession:<25} {'N/A':>10} {'N/A':>10}\n")
                continue

            kl, clip_score = parsed
            results.append((profession, kl, clip_score))

            line = f"{profession:<25} {kl:>10.6f} {clip_score:>10.4f}"
            print(f"  {line}")
            fout.write(f"{profession:<25} {kl:>10.6f} {clip_score:>10.4f}\n")

        # =====================================================
        # Averages
        # =====================================================
        if results:
            avg_kl   = sum(r[1] for r in results) / len(results)
            avg_clip = sum(r[2] for r in results) / len(results)

            fout.write(f"\n{'-'*50}\n")
            fout.write(f"{'AVERAGE':<25} {avg_kl:>10.6f} {avg_clip:>10.4f}\n")
            fout.write(f"{'='*50}\n")
            fout.write(f"Total professions evaluated: {len(results)}\n")

            print(f"\n{'='*50}")
            print(f"  {'AVERAGE':<23} KL={avg_kl:.6f}  CLIP={avg_clip:.4f}")
            print(f"  Professions evaluated: {len(results)}")
            print(f"  Results saved to: {out_file}")
            print(f"{'='*50}\n")
        else:
            print("  No valid results found.")
            fout.write("No valid results found.\n")