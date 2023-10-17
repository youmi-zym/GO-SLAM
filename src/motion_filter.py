import torch
import lietorch

from .geom import projective_ops as pops
from .modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """
    def __init__(self, net, video, thresh=2.5, device='cuda:0'):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        # image: [1, b, 3, h, w], net: [1, b, 128, h//8, w//8], inp: [1, b, 128, h//8, w//8]
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(dim=0), inp.relu().squeeze(dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ feature for correlation volume """
        # image: [1, b, 3, h, w], return: [1, b, 128, h//8, w//8]
        return self.fnet(image).squeeze(dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, timestamp, image, depth=None, intrinsic=None, gt_pose=None):
        """ main update operation - run on every frame in video """

        scale_factor = 8.0
        IdentityMat = lietorch.SE3.Identity(1, ).data.squeeze()

        batch, _, imh, imw = image.shape
        ht = imh // scale_factor
        wd = imw // scale_factor

        # normalize images, [b, 3, imh, imw] -> [1, b, 3, imh, imw], b=1 for mono, b=2 for stereo
        inputs = image.unsqueeze(dim=0).to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs) # [b, c, imh//8, imw//8]

        ### always add frist frame to the depth video ###
        left_idx = 0 # i.e., left image, for stereo case, we only store the hidden or input of left image
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:, [left_idx,]])  # [1, 128, imh//8, imw//8]
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(timestamp, image[left_idx], IdentityMat, 1.0, depth,
                              intrinsic/scale_factor, gmap, net[left_idx], inp[left_idx], gt_pose)

        ### only add new frame if there is enough motion ###
        else:
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]  # [1, 1, imh//8, imw//8, 2]
            corr = CorrBlock(self.fmap[None, [left_idx]], gmap[None, [left_idx]])(coords0)  # [1, 1, 4*49, imh//8, imw//8]

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)  # [1, 1, imh//8, imw//8, 2]

            # check motion magnitude / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:, [left_idx]])  # [1, 128, imh//8, imw//8]
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(timestamp, image[left_idx], None, None, depth,
                                  intrinsic/scale_factor, gmap, net[left_idx], inp[left_idx], gt_pose)

            else:
                self.count += 1