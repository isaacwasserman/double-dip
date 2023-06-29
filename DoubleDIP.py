import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tqdm
import numpy as np
import numba
from sklearn.neighbors import NearestNeighbors
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb, slic
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage.measurements import center_of_mass
from functools import partial
import matplotlib.pyplot as plt
import gc
from skimage.filters import threshold_otsu

# LeakyReLU activations
# Strided convolution downsampling
# Bilinear interpolation upsampling
# Reflection padding in convolutions

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, torch.nn.Conv2d): torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.Linear): m.weight.data.normal_(0, 0.01)
    for x in m.children(): init_cnn(x)


def group_norm(channels):
    ng = channels // 16 if not channels % 16 else channels // 4
    return torch.nn.GroupNorm(max(1, ng), channels)


def batch_norm(channels):
    return torch.nn.BatchNorm2d(channels)
    

def conv(inc, outc, ks=3, stride=1, norm=group_norm):
    c = torch.nn.Conv2d(inc, outc, ks, padding=0, stride=stride, bias=False)
    bn = norm(outc)
    a = torch.nn.LeakyReLU(inplace=True)
    mods = [c, bn, a]
    if ks > 1:
        p = torch.nn.ReflectionPad2d([ks // 2] * 4)
        mods = [p] + mods
    return torch.nn.Sequential(*mods)


class DownBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, norm=group_norm):
        super().__init__()
        self.layer1 = conv(inc, outc, ks, stride=2, norm=norm)
        self.layer2 = conv(outc, outc, ks, norm=norm)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, norm=group_norm):
        super().__init__()
        self.bn = norm(inc)
        self.layer1 = conv(inc, outc, ks)
        self.layer2 = conv(outc, outc, ks)

    def forward(self, x):
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class DIP(torch.nn.Module):
    def __init__(self, inc, c_down, k_down, c_up, k_up, c_skip=None,
                 k_skip=None, norm='group', c_out=3):
        super().__init__()
        assert len(c_down) == len(k_down)
        assert len(c_up) == len(k_up)
        assert len(c_down) == len(c_up)
        if c_skip is None:
            assert k_skip is None
        if k_skip is None:
            assert c_skip is None

        nblock = len(c_down)
        skip = c_skip is not None
        c_skip = [0]*nblock if c_skip is None else c_skip
        k_skip = [0]*nblock if k_skip is None else k_skip

        assert norm in ('group', 'batch')
        norm = globals()[norm+'_norm']

        inc = [inc] + c_down[:-1]
        down = []
        up = []
        for channels, ksize, ic in zip(c_down, k_down, inc):
            block = DownBlock(ic, channels, ksize, norm=norm)
            down.append(block)
        inc = [c_down[-1]] + c_up[:-1]
        for channels, ksize, ic, sk in zip(c_up, k_up, inc, c_skip):
            block = UpBlock(ic + sk, channels, ksize, norm=norm)
            up.append(block)

        if skip:
            skips = []
            for channels, ksize, dw in zip(c_skip, k_skip, c_down):
                block = conv(dw, channels, ksize, norm=norm)
                skips.append(block)

        self.down = torch.nn.ModuleList(down)
        self.up = torch.nn.ModuleList(up)
        self.skip = torch.nn.ModuleList(skips) if skip else None
        self.out_conv = torch.nn.Conv2d(c_up[-1], c_out, 1, bias=True)

        init_cnn(self)

    def forward(self, x):
        skips = []
        for i in range(len(self.down)):
            x = self.down[i](x)
            if self.skip is not None:
                skips.append(self.skip[i](x))

        for i in range(len(self.up)):
            if self.skip is not None:
                x = torch.cat((x, skips[-(i+1)]), 1)
            x = self.up[i](x)

        x = self.out_conv(x)
        x = torch.sigmoid(x)

        return x


class DoubleDIP(torch.nn.Module):
    def __init__(self, dip1, dip2, dipm):
        super().__init__()
        self.dip1 = dip1
        self.dip2 = dip2
        self.dipm = dipm

    def forward(self, z1, z2, zm):
        y1 = self.dip1(z1)
        y2 = self.dip2(z2)
        ym = self.dipm(zm)

        return y1, y2, ym


def dip(norm='group', c_out=3):
    return DIP(32, [128]*5, [3]*5, [128]*5, [3]*5, [4]*5, [1]*5,
               norm=norm, c_out=c_out)

def quantize(img, k=12):
    img = np.asarray(img)
    img = (img * (k/256)).astype(np.uint32)
    colors = (img * [k**2, k, 1]).sum(2)
    lut = np.stack([np.arange(k**3)//k**2,
                   (np.arange(k**3)//k)%k,
                   np.arange(k**3)%k], 1)
    hst, bins = np.histogram(colors, np.arange(k**3+1))
    ind = np.argsort(hst)[::-1]
    cdf = hst[ind].cumsum() / hst.sum()
    idx = np.searchsorted(cdf, 0.95, side='right') + 1
    ind = ind[:idx]
    colors = lut[bins[ind]].reshape(-1, 3)
    probs = hst[ind] / hst[ind].sum()
    ind = cdist(img.reshape(-1, 3), colors, 'sqeuclidean').argmin(1)
    ind = ind.reshape(img.shape[:2])
    colors = colors * int(256/k)
    colors = colors.astype(np.uint8)

    return ind, colors, probs


def smoothed_saliency(ind, colors, probs):
    lab = rgb2lab(colors[None].astype(np.uint8)).squeeze()
    #lab_dist = np.square(lab[...,None] - lab.T).sum(1)
    lab_dist = squareform(pdist(lab, 'sqeuclidean'))
    s = (lab_dist * probs).sum(1)
    s = (s - s.min()) / (s.max() - s.min())
    m = lab.shape[0] // 4
    dist, nn = NearestNeighbors(m).fit(lab).kneighbors()
    T = dist.sum(1)
    sp = ((T[:,None] - dist) * s[nn]).sum(1) / ((m-1)*T)

    return sp


def hc_saliency(img):
    ind, colors, probs = quantize(img)
    sal = smoothed_saliency(ind, colors, probs)
    sal = (sal - sal.min()) / (sal.max() - sal.min())
    sal_img = sal[ind]
    return sal_img


@numba.njit(parallel=True)
def region_saliency(lab_dist, histo, reg_sizes, reg_dist, v=0.4):
    n, k = histo.shape
    S = np.zeros(n)
    for i in numba.prange(n):
        for j in range(i+1, n):
            d = 0
            for c1 in range(k):
                for c2 in range(k):
                    d += histo[i, c1] * histo[j, c2] * lab_dist[c1, c2]
            w = np.exp(-reg_dist[i, j] / v)
            S[i] += (w * reg_sizes[j] * d)
            S[j] += (w * reg_sizes[i] * d)
    return S


def rc_saliency(img):
    ind, colors, probs = quantize(img)
    lab = rgb2lab(colors[None].astype(np.uint8)).squeeze()
    lab_dist = squareform(pdist(lab, 'sqeuclidean'))
    # region segmentation
    #regions = felzenszwalb(np.asarray(img), max(img.size)/2)
    regions = slic(np.asarray(img), 200, slic_zero=True)
    # region histograms
    histo = np.zeros((regions.max()+1, colors.shape[0]))
    np.add.at(histo, (regions, ind), 1)
    reg_sizes = np.bincount(regions.ravel())
    # region centroid distances
    centroids = center_of_mass(regions+1, regions, np.arange(regions.max()+1))
    centroids = np.array(centroids) / ind.shape
    reg_dist = squareform(pdist(centroids, 'euclidean'))

    reg_sal = region_saliency(lab_dist, histo, reg_sizes, reg_dist)
    reg_sal /= reg_sal.max()
    sal_img = reg_sal[regions]

    return sal_img

subplots = partial(plt.subplots, constrained_layout=True)

class SobelGradient(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = [[1., 2, 1], [0, 0, 0], [-1, -2, -1]]
        fy = torch.tensor(weights, requires_grad=False)
        fx = torch.tensor(weights, requires_grad=False).t()
        filter = torch.stack((fx, fy), 0).view(2, 1, 3, 3)
        self.register_buffer('filter', filter)
        
    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        g = F.conv2d(x, self.filter, padding=1).pow(2).sum(dim=1, keepdim=True).sqrt()
        return g
    
class ExclusionLoss_catalys1(torch.nn.Module):
    interp = partial(F.interpolate, align_corners=False)
    def __init__(self, n=3):
        super().__init__()
        self.sobel = SobelGradient()
        self.n = n
        
    def psi(self, x, y):
        xg = self.sobel(x)
        yg = self.sobel(y)
        
        lamx = yg.norm(dim=(2, 3)).div(xg.norm(dim=(2, 3))).sqrt()[...,None,None]
        lamy = 1 / lamx
        
        vx = xg.abs().mul(lamx).tanh()
        vy = yg.abs().mul(lamy).tanh()
        
        return vx * vy
        
    def forward(self, x1, x2):
        loss = 0
        for i in range(self.n):
            if i > 0:
                x1 = self.interp(x1, scale_factor=0.5, mode='bilinear')
                x2 = self.interp(x2, scale_factor=0.5, mode='bilinear')
            loss = loss + self.psi(x1, x2).norm(dim=(2, 3)).mean()
        return loss
    
class ExclusionLoss_Official(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss_Official, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady
    
class SegMaskRegLoss(torch.nn.Module):
    def forward(self, mask):
        loss = 1 / (mask - 0.5).abs().sum(dim=(-1, -2))
        loss = loss.mean()
        return loss
    
class DoubleDipSegmentation(object):
    def __init__(self, img, device='cuda'):
        # DoubleDip network
        dip_ = partial(dip, norm='batch')
        self.net = DoubleDIP(dip_(), dip_(), dip_(c_out=1))
        # loss functions
        self.recon_loss = torch.nn.L1Loss()
        self.excl_loss = ExclusionLoss_Official()
        self.reg_loss = SegMaskRegLoss()
        self.device = device
        self._init_data(np.array(img))
    
    def _reset_metrics(self):
        self.metrics = {'recon.loss':[],
                        'excl.loss': [],
                        'reg.loss': []}
    
    @staticmethod
    def _augmented_noise_image(h, w, device):
        z = torch.rand(32, h, w).to(device) * 0.1 - 0.05
        z = [z, z.flip(-1).transpose(-1,-2), z.flip((-1,-2)), z.transpose(-1,-2).flip(-1)]
        z += [x.flip(i) for x, i in zip(z, (-1, -2, -1, -2))]
        z = torch.stack(z, 0)
        z.requires_grad = False
        return z
    
    def _get_saliency(self, img):
        small = img.resize([x//2 for x in img.size], Image.BILINEAR)
        sal = hc_saliency(small)
        sal = np.asarray(Image.fromarray(sal).resize(img.size, Image.BILINEAR))
        return sal
        
    def _init_data(self, img_):
        longest_side = max(img_.shape[:2])
        while not ((longest_side % np.array([1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224])) == 0).all():
            longest_side += 1
        longest_side = 224
        img = Image.fromarray(img_).resize((longest_side, longest_side), Image.BILINEAR)
        w, h = img.size
        # img + augmentations
        flip = img.transpose(0)
        imgs = ([img] + [img.transpose(i) for i in range(2, 5)] +
                [flip] + [flip.transpose(i) for i in range(2, 5)])

        imgs = np.stack([np.asarray(x) for x in imgs], 0)
        imgs = torch.from_numpy(imgs).permute(0,3,1,2)
        imgs = imgs.float().div(255)
        # noise input + augmentations
        za = self._augmented_noise_image(h, w, self.device)
        zb = self._augmented_noise_image(h, w, self.device)
        zc = self._augmented_noise_image(h, w, self.device)
        # saliency-based segmentation hints
        sal = self._get_saliency(img)
        hint = Image.fromarray(255 * (sal > 0.75).astype(np.uint8))
        flip = hint.transpose(0)
        hints = ([hint] + [hint.transpose(i) for i in range(2, 5)] +
                [flip] + [flip.transpose(i) for i in range(2, 5)])
        hints = torch.from_numpy(np.stack([np.asarray(x) for x in imgs], 0))[:,None,...]
        hints = hints.float().div(255)
        
        self.imgs = imgs
        self.hints = hints
        self.za = za
        self.zb = zb
        self.zc = zc
        self.noise_map = torch.empty_like(za)
    
    def _add_noise(self, z, i, std=0.02):
#         noise = torch.normal(0.0, std * math.log10(i+1), z.shape).cuda()
        self.noise_map.normal_(0.0, max(.01, std * np.log10(i+1)))
        return z + self.noise_map
    
    def _hint_loss(self, y1, y2, m):
        n = F.l1_loss(m, torch.zeros_like(m))
        return F.l1_loss(m*y1, m*y2).div(n)
        
    def _compute_loss(self, y1, y2, m, iters):
        # We may want to add different weights to each loss. They could be on different
        # scales. Also, the gradients might be on different scales.
        mixed = m * y1 + (1 - m) * y2
        recon_loss = self.recon_loss(self.imgs, mixed)
        excl_loss = self.excl_loss(y1, y2)
        reg_loss = self.reg_loss(m)
        total_loss = recon_loss + reg_loss + excl_loss
        if iters < 200:
            total_loss = (total_loss + 
                          self._hint_loss(y1, y2, self.hints) +
                          self._hint_loss(y1, y2, 1-self.hints))
        
        self.metrics['recon.loss'].append(recon_loss.item())
        self.metrics['excl.loss'].append(excl_loss.item())
        self.metrics['reg.loss'].append(reg_loss.item())
        
        return total_loss
        
    def fit(self, iters=1000, z_std=0.02, show_progress=True):
        self._reset_metrics()
        iterator = range(iters)
        if show_progress:
            iterator = tqdm.tqdm(iterator, total=iters)
            
        optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        inputs = (self.za, self.zb, self.zc)
        
        for i in iterator:
            za, zb, zc = (self._add_noise(z, i, z_std).to(self.imgs.device) for z in inputs)
            y1, y2, m = self.net(za, zb, zc)
            loss = self._compute_loss(y1, y2, m, i)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if show_progress:
                iterator.set_postfix_str(str(loss.item()))
    
    @torch.no_grad()
    def reconstruct(self):
        y1, y2, m = self.net(self.za, self.zb, self.zc)
        res = m * y1 + (1 - m) * y2
        return res
    
    @torch.no_grad()
    def layers(self):
        y1, y2, m = self.net(self.za, self.zb, self.zc)
        return y1, y2, m
    
    def to(self, device):
        self.net = self.net.to(device)
        self.imgs = self.imgs.to(device)
        self.hints = self.hints.to(device)
        self.excl_loss = self.excl_loss.to(device)
        return self
    
def segment(img, iters=1000, device='cuda'):
    segmentor = DoubleDipSegmentation(img, device=device).to(device)
    segmentor.fit(iters)
    res = segmentor.reconstruct()
    with torch.no_grad():
        y1, y2, m = segmentor.net(segmentor.za, segmentor.zb, segmentor.zc)
    del segmentor
    torch.cuda.empty_cache()
    gc.collect()
    return m.cpu().numpy()

if __name__ == "__main__":
    # accept command line arguments
    import argparse
    import glob
    import os
    import pathlib
    parser = argparse.ArgumentParser(description='DoubleDIP Segmentation')
    parser.add_argument('--input', type=str, default='images', help='input image')
    parser.add_argument('--output', type=str, default='output', help='output image')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=-1, help='end index')
    input_dir = parser.parse_args().input
    output_dir = parser.parse_args().output
    start_ind = parser.parse_args().start
    end_ind = parser.parse_args().end
    if end_ind == -1:
        end_ind = len(glob.glob(os.path.join(input_dir, '*')))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i,img_path in enumerate(glob.glob(os.path.join(input_dir, '*.jpg'))[start_ind:end_ind]):
        if os.path.exists(os.path.join(output_dir, os.path.basename(img_path))):
            continue
        else:
            p = pathlib.Path(os.path.join(output_dir, os.path.basename(img_path)))
            try:
                p.touch(exist_ok=False)
            except:
                print(f"output file {os.path.join(output_dir, os.path.basename(img_path))} already exists. skipping...")
                continue
            print(f"Segmenting image {i+1}\n")
            img = plt.imread(img_path)
            probs = segment(img, iters=1000)[0,0]
            threshold = threshold_otsu(probs)
            mask = probs > threshold
            mask = Image.fromarray((255*mask).astype(np.uint8))
            mask.save(os.path.join(output_dir, os.path.basename(img_path)))
            num_in_dir = len(list(glob.glob(os.path.join(output_dir, '*.jpg'))))
            print(num_in_dir, "images segmented")
