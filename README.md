# Spk-Res (Coming soon)
## Install
You can create the environment using a yml file:
```
conda env create -f environment.yml
```
You can also manually install the packages:
```
conda create -n spk_res python=3.9
conda activate spk_res
pip install matplotlib scikit-learn yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb pyyaml requests scipy tb-nightly yapf lpips
pip install spikeinterface[full]==0.100.6
pip install pynwb==2.5.0
pip install kilosort
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install einops
pip install numpy==1.26.4
pip install scikit-image==0.22.0
pip install opencv-python-headless
pip install spikeextractors==0.9.11
pip install spikecomparison==0.3.3
pip install hdmf==3.14.0
```
## Configure
You can set parameters by configuring the yml file in the options_linux folder.
