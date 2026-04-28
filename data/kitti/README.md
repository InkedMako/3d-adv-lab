# KITTI Data Placeholder

Place KITTI files into this folder using the standard structure.

Required for training metadata generation:
- ImageSets/train.txt
- ImageSets/val.txt
- training/image_2/*.png
- training/velodyne/*.bin
- training/calib/*.txt
- training/label_2/*.txt

After copying files, run:

python scripts/prepare_kitti.py --data-root data/kitti
