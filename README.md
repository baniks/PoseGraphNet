# PoseGraphNet
This is the implementation of the PoseGraphNet model proposed in the paper:

>Banik, Soubarna, Alejandro Mendoza GarcÍa, and Alois Knoll. "3D human pose regression using graph convolutional network." 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021.

## Results on Human3.6M

| Model  | MPJPE (P1) |
| ------------- | ------------- |
| PoseGraphNet (Mask R-CNN)   | 59.5  |
| PoseGraphNet (CPN)  | 52.8  |


## Dataset Human3.6M

Download Human3.6M from http://vision.imar.ro/human3.6m/ into data_dir. The directory structure should look like this
```
├── S1
├── S11
├── S5
├── S6
├── S7
├── S8
└── S9
    └── MyPoseFeatures
        ├── D2_Positions
        ├── D3_Positions
        └── D3_Positions_mono
```

## Donwload Mask R-CNN and CPN detections
Refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md#mask-r-cnn-and-cpn-detections)
```
cd data
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
```

## To train the model
### Using CPN Predicted 2D input
```
python train_posegraphnet_singleloss.py --exp='experiment' --exp_suffix='run1' --run_suffix=1 --exp_desc="description" --data_dir='<DATA_DIR>' --cpn_file='<PATH TO CPN PREDICTED 2D POSE>'
```

### Using ground truth 2D input
set ds_category='gt' in params.json in experiment directory
```
python train_posegraphnet_singleloss.py --exp='experiment' --exp_suffix='run1' --run_suffix=1 --exp_desc="description" --data_dir='<DATA_DIR>'
```

## To evaluate the model
```
python train_posegraphnet_singleloss.py --exp='icip_v2' --exp_suffix='run3' --run_suffix='1' --exp_desc='evaluate icip_v2/run3' --test --checkpoint='../models/icip_v2/run3/best.pth.tar' --data_dir='<DATA_DIR>' --cpn_file='<PATH TO CPN PREDICTED 2D POSE>'
```

## Citation
If you use our code, please cite as follow:

>@inproceedings{banik20213d,
  title={3D human pose regression using graph convolutional network},
  author={Banik, Soubarna and Garc{\'I}a, Alejandro Mendoza and Knoll, Alois},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={924--928},
  year={2021},
  organization={IEEE}
}
