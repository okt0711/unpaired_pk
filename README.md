# Unpaired deep learning for pharmacokinetic parameter estimation from dynamic contrast-enhanced MRI without AIF measurements

This repository is the official pytorch implementation of "Unpaired deep learning for pharmacokinetic parameter estimation from dynamic contrast-enhanced MRI without AIF measurements
".

> "Unpaired deep learning for pharmacokinetic parameter estimation from dynamic contrast-enhanced MRI without AIF measurements
",  
> Gyutaek Oh, Yeonsil Moon, Won-Jin Moon, and Jong Chul Ye,  
> NeuroImage, 2024 [[Paper]](https://www.sciencedirect.com/science/article/pii/S1053811924000661)

## Requirements
The code is implented in Python 3.7 with below packages.
```
torch               1.8.1
numpy               1.21.6
scipy               1.7.3
```

## Training and Inference
To evaluate, run the below commands.
```
sh run_cycle_tumor.sh
sh run_cycle_mci.sh
```
To train the model, add the ```--training``` option in the script files.
We also provide source codes for baseline supervised methods, and you can run them with ```run_supervised_tumor.sh```, ```run_supervised_mci.sh```, ```run_supervised_physics_tumor.sh```, and ```run_supervised_physics_mci.sh```.

## Citation
If you find our work interesting, please consider citing
```
@article{oh2024unpaired,
  title={Unpaired deep learning for pharmacokinetic parameter estimation from dynamic contrast-enhanced MRI without AIF measurements},
  author={Oh, Gyutaek and Moon, Yeonsil and Moon, Won-Jin and Ye, Jong Chul},
  journal={NeuroImage},
  volume={291},
  pages={120571},
  year={2024},
  publisher={Elsevier}
}
```
