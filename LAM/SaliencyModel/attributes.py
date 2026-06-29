import torch
import cv2
import torch.nn.functional as F


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def reduce_func(method):
    """
    
    :param method: ['mean', 'sum', 'max', 'min', 'count', 'std']
    :return:
    """
    if method == 'sum':
        return torch.sum
    elif method == 'mean':
        return torch.mean
    elif method == 'count':
        return lambda x: sum(x.size())
    else:
        raise NotImplementedError()


def attr_id(tensor, h, w,window_h=1,window_w=1, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    crop = tensor[:, :, h: h + window_h, w: w + window_w]
    return reduce_func(reduce)(crop)

# 差分算子
def attr_grad(tensor, h=None, w=None,window_h=1,window_w=1,reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2) # 一阶差分算子
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    if h==None or w==None:
        crop = grad
    else:
        crop = grad[:, :, h: h + window_h, w: w + window_w]
    return reduce_func(reduce)(crop) # 求和


# gabor_filter = cv2.getGaborKernel((21, 21), 10.0, -np.pi/4, 8.0, 1, 0, ktype=cv2.CV_32F)

def attr_gabor_generator(gabor_filter):
    filter = torch.from_numpy(gabor_filter).view((1, 1,) + gabor_filter.shape).repeat(1,3,1,1)
    def attr_gabor(tensor, h, w, window=8, reduce='sum'):
        after_filter = F.conv2d(tensor, filter, bias=None)
        crop = after_filter[:, :, h: h + window, w: w + window]
        return reduce_func(reduce)(crop)
    return attr_gabor


