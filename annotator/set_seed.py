import argparse
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
SEED_FILES = {
    "round1": BASE_DIR / "annotation_seed.txt",
    "relevance": BASE_DIR / "reannotation" / "relevance" / "annotation_seed.txt",
    "flags": BASE_DIR / "reannotation" / "flags" / "annotation_seed.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", type=int, nargs="?", help="Seed to use (random if omitted)")
    parser.add_argument("--app", choices=SEED_FILES, default="round1", help="Which app's seed to set (default: round1, i.e. app.py)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing seed file")
    args = parser.parse_args()

    seed_file = SEED_FILES[args.app]
    name = seed_file.relative_to(BASE_DIR)
    if seed_file.exists() and not args.force:
        existing = seed_file.read_text().strip()
        raise SystemExit(
            f"{name} already exists (seed={existing}). Refusing to overwrite -- "
            f"changing it now would reshuffle the walk order for anyone mid-annotation. "
            f"Pass --force if you really mean to reset it."
        )

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    seed_file.write_text(f"{seed}\n")
    print(f"Seed {seed} written to {name}")


if __name__ == "__main__":
    main()
