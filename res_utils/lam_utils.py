import numpy as np
import torch
from matplotlib import pyplot as plt
# 设置字体和字号
from matplotlib.colors import LinearSegmentedColormap
from test_LAM.ModelZoo.utils import _add_batch_one
from test_LAM.SaliencyModel.attributes import attr_grad

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

# 设置color
main_color = '#233342'
good_color = '#6A8F95'
bad_color = '#B8A49D'
di_color = '#B78C7C'
cmap = LinearSegmentedColormap.from_list('custom', [(0, 'white'), (1, 'red')])

def get_closest_channels(recording, channel_ids=None, num_channels=None):
    """Get closest channels + distances

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to get closest channels
    channel_ids: list
        List of channels ids to compute there near neighborhood
    num_channels: int, default: None
        Maximum number of neighborhood channels to return

    Returns
    -------
    closest_channels_inds : array (2d)
        Closest channel indices in ascending order for each channel id given in input
    dists: array (2d)
        Distance in ascending order for each channel id given in input
    """
    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    if num_channels is None:
        num_channels = len(channel_ids) - 1

    locations = recording.get_channel_locations(channel_ids=channel_ids)

    closest_channels_inds = []
    dists = []
    for i in range(locations.shape[0]):
        distances = np.linalg.norm(locations[i, :] - locations, axis=1)
        order = np.argsort(distances)
        closest_channels_inds.append(order[0 : num_channels + 1])
        dists.append(distances[order][0 : num_channels + 1])

    return np.array(closest_channels_inds), np.array(dists)


# 2. 分析最重要的像素
def plot_input_weight(slt_intp,slt_grad,neigh_chls,mcl,bad_chans,locations,figsize=(2, 2)):
    xs = np.linspace(-5, 5, slt_intp[-1].shape[0])
    fig, ax = plt.subplots(1, 1, figsize=figsize,constrained_layout=True)
    for i, wfs_i in enumerate(slt_intp[-1].T):  # 输入数据
        chl_i = neigh_chls[i]
        if chl_i == mcl:
            color = main_color  # 需要插值的通道
        elif np.isin(chl_i, bad_chans):
            color = bad_color  # 坏通道
        else:
            color = good_color  # 好通道

        plt.plot(locations[i][0] + xs, locations[i][1] + wfs_i * 0.045 + 12, color=color)
        ax.scatter(locations[i][0], locations[i][1], s=30, marker='s', color=color)
        text = ax.annotate('C{}'.format(neigh_chls[i]), (locations[i][0]-1.2, locations[i][1] + 2.5), color=color)
        text.set_fontsize(8)

        # 积分因子
        grad_i = slt_grad[:, i]
        ax.scatter(locations[i][0] + xs, locations[i][1] + wfs_i * 0.045 + 12, s=20, marker='o', color='red',alpha=grad_i)

        # 标注振幅
        text_amp = ax.annotate(int(np.min(wfs_i)), (locations[i][0] + 2.5, locations[i][1] - 16), color=color)
        text_amp.set_fontsize(8)

    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def plot_intp_image(slt_intp,neigh_chls,mcl,bad_chans,figsize=(6, 1.5),need_color=False):
    # sharey：共享坐标轴 / constrained_layout：自行调整子图
    fig, axes = plt.subplots(1, slt_intp.shape[0], figsize=figsize, sharey=True, constrained_layout=True)
    xs = np.linspace(-5, 5, slt_intp.shape[1])
    ys = np.linspace(-50, 50, slt_intp.shape[2])
    for i, intp_i in enumerate(slt_intp):
        for j, wfs_j in enumerate(intp_i.T):
            if need_color:
                chl_j = neigh_chls[j]
                if chl_j == mcl:
                    color = main_color  # 需要插值的通道
                elif np.isin(chl_j, bad_chans):
                    color = bad_color  # 坏通道
                else:
                    color = good_color  # 好通道
            else:
                color = 'black'
            axes[i].plot(xs, wfs_j * 0.045 + ys[j], color=color)
            axes[i].annotate(int(np.min(wfs_j)), (-2, np.mean(wfs_j) * 0.045 + ys[j] - 15), color=color)
        if i == 0:
            # 设置 y 轴标签
            y_labels = [f'C{chl}' for chl in neigh_chls]
            axes[i].set_yticks(ys - 15)
            axes[i].set_yticklabels(y_labels)
            axes[i].get_yaxis().set_visible(True)
        else:
            axes[i].get_yaxis().set_visible(False)
        axes[i].set_xticks([])
    return fig


def plot_target(target_numpy,figsize=(5, 1)):
    fig, ax = plt.subplots(1, 1, figsize=figsize,constrained_layout=True)
    ax.plot(target_numpy,color='black')
    ax.set_xticks([]) # 设置 x 轴标签
    ax.set_yticks([]) # 设置 y 轴标签
    return fig


def plot_magnitude_of_gradient(lambda_derivative_interpolation,fold,figsize=(5, 1)):
    mgnt_grad_arr = np.zeros([fold])
    for i in range(fold):
        ldi=lambda_derivative_interpolation[i]
        mgnt_grad = attr_grad(_add_batch_one(torch.from_numpy(ldi)).cuda()) # 计算梯度幅值
        mgnt_grad_arr[i] = mgnt_grad

    fig, ax = plt.subplots(1, 1, figsize=figsize,constrained_layout=True)
    ax.plot(mgnt_grad_arr,color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig