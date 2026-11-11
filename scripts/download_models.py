"""Download default models for ADAS Road Assistant."""
import sys
from pathlib import Path


def download_yolo26n():
    """Download the standard YOLO26n COCO model."""
    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    target = models_dir / 'yolo26n.pt'

    if target.exists():
        print(f'Already exists: {target}')
        return

    try:
        from ultralytics import YOLO

        print('Downloading YOLO26n (COCO pretrained)...')
        model = YOLO('yolo26n.pt')
        # ultralytics downloads to current dir; move it
        downloaded = Path('yolo26n.pt')
        if downloaded.exists():
            downloaded.rename(target)
        print(f'Saved to {target}')
    except ImportError:
        print(
            'ERROR: ultralytics not installed. Run: pip install ultralytics',
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    print('=== ADAS Model Downloader ===')
    download_yolo26n()
    print('\nNote: Traffic sign detection model (traffic_signs_yolo26n.pt) requires')
    print('a custom-trained model. See README for training instructions.')
    print('Done.')


if __name__ == '__main__':
    main()
