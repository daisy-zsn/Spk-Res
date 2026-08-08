# Spk-Res
## Introdcution
尝试用深度学习方法修复高密度阵列电极坏道上的数据
高密度阵列电极采集的信号会在相邻电极上留下印迹，信号间存在一种空间关系，一般插补坏道用kriging方法。
自然的想到用kriging方法是否可以捕捉这种非线性空间关系，尤其当主通道的信号缺失时。
另外，用LAM尝试理解模型学到了什么东西。

## Dataset

## Install
You can create the environment:
```
conda create -n spk_res python=3.9
conda activate spk_res
pip install -r requirements.txt
```


## Configure
You can set parameters by configuring the yml file in the options_linux folder.

## Experiments

1. nwb file

2. bin file

3. mea file


## 存在的问题
1. 
2. MAE
3. 因为我水平过于低下并且没钱租充足的显卡，所以很多优化没做好，sorry


