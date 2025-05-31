# Hướng dẫn Download Datasets

## PASCAL VOC 2012

### Option 1: Manual Download
```bash
cd data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

### Option 2: Using torchvision
```python
import torchvision.datasets as datasets

# Download automatically
voc_train = datasets.VOCSegmentation(
    root='./data', 
    year='2012', 
    image_set='train', 
    download=True
)
```

## Cityscapes Dataset

1. Register at: https://www.cityscapes-dataset.com/
2. Download:
   - leftImg8bit_trainvaltest.zip (11GB)
   - gtFine_trainvaltest.zip (241MB)
3. Extract to data/cityscapes/

## ADE20K Dataset

```bash
cd data/
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

## Quick Start với Sample Data

Nếu bạn chỉ muốn test nhanh:
```bash
python scripts/setup_demo.py
python scripts/test_models.py
python scripts/quick_demo.py
```
