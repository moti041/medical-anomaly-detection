import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def parse_args():
    parser = argparse.ArgumentParser(description='Create val/NORMAL from train/NORMAL deterministically.')
    parser.add_argument('--data_dir', default='data/chest_xray', help='Dataset root directory')
    parser.add_argument('--ratio', type=float, default=0.15, help='Fraction of train/NORMAL to copy into val/NORMAL')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic selection')
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.ratio < 1:
        raise ValueError('--ratio must be between 0 and 1')

    data_dir = Path(args.data_dir)
    train_normal = data_dir / 'train' / 'NORMAL'
    val_normal = data_dir / 'val' / 'NORMAL'
    if not train_normal.exists():
        raise FileNotFoundError(f'Missing train/NORMAL directory: {train_normal}')

    val_normal.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in train_normal.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f'No images found in {train_normal}')

    target_count = max(1, int(round(len(images) * args.ratio)))
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(images, target_count), key=lambda path: path.name)

    copied = 0
    for src in selected:
        dst = val_normal / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    manifest = data_dir / 'val' / 'normal_manifest.txt'
    manifest.write_text('\n'.join(path.name for path in selected) + '\n', encoding='utf-8')

    print(f'train/NORMAL images: {len(images)}')
    print(f'val/NORMAL selected: {len(selected)}')
    print(f'new files copied: {copied}')
    print(f'val directory: {val_normal}')
    print(f'manifest: {manifest}')
    print('Note: future training excludes val filenames from train/NORMAL in ChestXrayDataset.')


if __name__ == '__main__':
    main()
