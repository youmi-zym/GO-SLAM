import torch
import numpy as np
from copy import deepcopy

from .modules import CorrBlock, AltCorrBlock
from .geom import projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device='cuda:0', corr_impl='volume', max_factors=-1.0, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        ht = video.ht // 8
        wd = video.wd // 8
        self.ht = ht
        self.wd = wd

        self.coords0 = pops.coords_grid(ht, wd, device)  # [ht, wd, 2]
        self.ii = torch.tensor([], dtype=torch.long, device=device)
        self.jj = torch.tensor([], dtype=torch.long, device=device)
        self.age = torch.tensor([], dtype=torch.long, device=device)  # [buffer, ht, wd]

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)],
        )

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0, 2, 3, 4], keepdim=False).cpu().numpy()
        w = w[ix]
        msg = 'INFO: Edges of Graph: \n Start  End    Weight\n'
        for e in zip(ii, jj, w):
            msg += f' {e[0]:05d}, {e[1]:05d}, {e[2]:.4f}\n'
        print(msg)

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0, 2, 3, 4], keepdim=False)
        mask = (torch.abs(self.ii - self.jj) > 2) & (conf < 1e-3)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]], dim=0)
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]], dim=0)
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """
        if not isinstance(ii, torch.Tensor):
            ii = torch.tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
            and self.corr is not None and remove:
            ix = torch.arange(len(self.age))[torch.argsort(self.age, descending=False).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(dim=0)

        # correlation volume for new edges
        if self.corr_impl == 'volume':
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(dim=0) # [1, num, channel, height, width]
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(dim=0)
            corr = CorrBlock(fmap1, fmap2)  # corr pyramid [num, h, w, h//{1,2,4,8}, w//{1,2,4,8}]
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(dim=0) # [1, num, channel, height, width]
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)  # [1, num, height, width, 2]
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], dim=0)  # [n, ]
        self.jj = torch.cat([self.jj, jj], dim=0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], dim=0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], dim=1)

        self.target = torch.cat([self.target, target], dim=1)
        self.weight = torch.cat([self.weight, weight], dim=1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], dim=0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], dim=0)
            self.target_inac = torch.cat([self.target_inac, self.target[:, mask]], dim=1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:, mask]], dim=1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == 'volume':
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:, ~mask]

        if self.inp is not None:
            self.inp = self.inp[:, ~mask]

        self.target = self.target[:, ~mask]
        self.weight = self.weight[:, ~mask]

    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]


        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:, ~m]
            self.weight_inac = self.weight_inac[:, ~m]

        m = (self.ii == ix) | (self.jj == ix)
        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)

    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, iters=2, use_inactive=False, EPS=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            # [batch, N, h, w, 2]
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motion = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = self.update_op(
            self.net, self.inp, corr, motion, self.ii, self.jj
        )

        if t0 is None:  # the first keyframe (i.e., 0) should be fixed
            t0 = max(1, self.ii.min().item() + 1)
        t0 = max(1, t0)
        if t1 is None:
            t1 = max(self.ii.max().item(), self.jj.max().item()) + 1

        with torch.cuda.amp.autocast(enabled=True):
            self.target = coords1 + delta.float()
            self.weight = weight.float()

            ht, wd, _ = self.coords0.shape
            self.damping[torch.unique(self.ii, sorted=True)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], dim=0)
                jj = torch.cat([self.jj_inac[m], self.jj], dim=0)
                target = torch.cat([self.target_inac[:, m], self.target], dim=1)
                weight = torch.cat([self.weight_inac[:, m], self.weight], dim=1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            damping_index = torch.arange(t0, t1).long().to(ii.device)
            damping_index = torch.unique(torch.cat([damping_index, ii], dim=0), sorted=True)
            damping = 0.2 * self.damping[damping_index].contiguous() + EPS

            target = target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            self.video.ba(target, weight, damping, ii, jj, t0=t0, t1=t1,
                          iters=iters, lm=1e-4, ep=0.1, motion_only=motion_only, ba_type=None)

            if self.upsample:
                self.video.upsample(torch.unique(self.ii, sorted=True), upmask)

        self.age += 1

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, iters=2, use_inactive=False, EPS=1e-7, steps=8, max_t=None, ba_type='dense', motion_only=False):
        """ run update operator on factor graph - reduced memory implementation """
        cur_t = self.video.counter.value

        # alternate corr implementation
        t = max_t if max_t is not None else cur_t

        sel_index = torch.arange(0, cur_t+2)
        num, rig, ch, ht, wd = self.video.fmaps[sel_index].shape  # rig = 1(mono); 2(stereo)
        corr_op = AltCorrBlock(self.video.fmaps[sel_index].view(1, num*rig, ch, ht, wd))

        if t0 is None:  # the first keyframe (i.e., 0) should be fixed
            t0 = max(1, self.ii.min().item() + 1)
        t0 = max(1, t0)
        if t1 is None:
            t1 = max(self.ii.max().item(), self.jj.max().item()) + 1

        for step in range(steps):
            with torch.cuda.amp.autocast(enabled=False):
                # [batch, N, h, w, 2]
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motion = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            s = 13
            for i in range(self.ii.min(), self.ii.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                if v.sum() < 1:
                    continue

                iis = self.ii[v]
                jjs = self.jj[v]

                # for stereo case, i.e., rig=2, each video.fmaps contain both left and right feature maps
                # edge ii == jj means stereo pair, corr1: [B, N, (2r+1)^2 * num_levels, H, W]
                corr1 = corr_op(coords1[:, v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                    # [B, N, C, H, W], [B, N, H, W, 2],[B, N, H, W, 2], [B, s, H, W], [B, s, 8*9*9, H, W]
                    net, delta, weight, damping, upmask = \
                    self.update_op(self.net[:, v], self.video.inps[None, iis], corr1, motion[:, v], iis, jjs)

                    if self.upsample:
                        self.video.upsample(torch.unique(iis, sorted=True), upmask)

                self.net[:, v] = net
                self.target[:, v] = coords1[:, v] + delta.float()
                self.weight[:, v] = weight.float()
                self.damping[torch.unique(iis, sorted=True)] = damping

            damping_index = torch.arange(t0, t1).long().to(self.ii.device)
            damping_index = torch.unique(torch.cat([damping_index, self.ii], dim=0), sorted=True)
            damping = 0.2 * self.damping[damping_index].contiguous() + EPS

            target = self.target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            # dense bundle adjustment, fix the first keyframe, while optimize within [1, t]
            if ba_type == 'loop':
                self.video.ba(target, weight, damping, self.ii, self.jj, t0=t0, t1=t1,
                              iters=iters, lm=1e-4, ep=1e-1, motion_only=motion_only, ba_type=ba_type)
            else:
                self.video.ba(target, weight, damping, self.ii, self.jj, t0=t0, t1=t1,
                              iters=iters, lm=1e-5, ep=1e-2, motion_only=motion_only, ba_type=ba_type)

            # for visualization
            self.video.dirty[:t] = True

    @torch.cuda.amp.autocast(enabled=True)
    def update_fast(self, t0=None, t1=None, iters=2, use_inactive=False, EPS=1e-7, steps=8, max_t=None, ba_type='loop', motion_only=False):
        """ run update operator on factor graph """
        if t0 is None:  # the first keyframe (i.e., 0) should be fixed
            t0 = max(1, self.ii.min().item() + 1)
        t0 = max(1, t0)
        if t1 is None:
            t1 = max(self.ii.max().item(), self.jj.max().item()) + 1

        for step in range(steps):

            # motion features
            with torch.cuda.amp.autocast(enabled=False):
                # [batch, N, h, w, 2]
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motion = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            # correlation features
            corr = self.corr(coords1)

            with torch.cuda.amp.autocast(enabled=True):
                self.net, delta, weight, damping, upmask = self.update_op(
                    self.net, self.inp, corr, motion, self.ii, self.jj
                )

            self.target = coords1 + delta.float()
            self.weight = weight.float()

            ht, wd, _ = self.coords0.shape
            self.damping[torch.unique(self.ii, sorted=True)] = damping

            damping_index = torch.arange(t0, t1).long().to(self.ii.device)
            damping_index = torch.unique(torch.cat([damping_index, self.ii], dim=0), sorted=True)
            damping = 0.2 * self.damping[damping_index].contiguous() + EPS

            target = self.target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            self.video.ba(target, weight, damping, self.ii, self.jj, t0=t0, t1=t1,
                          iters=iters, lm=1e-4, ep=1e-1, motion_only=motion_only, ba_type=ba_type)

            if self.upsample:
                self.video.upsample(torch.unique(self.ii, sorted=True), upmask)

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """
        ii, jj = torch.meshgrid(
            torch.arange(t0, t1),
            torch.arange(t0, t1),
            indexing='ij',
        )

        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0
        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)

        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False, max_t=None):
        """ add edges to the factor graph based on distance """

        t = max_t if max_t is not None else self.video.counter.value
        ilen, jlen = t-t0, t-t1
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf
        d = d.reshape(ilen, jlen)

        # filter out these edges which had been built before
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], dim=0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], dim=0)

        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            if (t0 <= i < t) and (t1 <= j < t):
                di, dj = i-t0, j-t1
                d[di, dj] = np.inf
                d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        es = []
        # build edges within local window [i-rad-1, i]
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                di, dj = i-t0, i-t1
                d[di, dj] = np.inf

            for j in range(max(i-rad, 0), i):
                es.append((i, j))
                es.append((j, i))
                di, dj = i-t0, j-t1
                d[di, dj] = np.inf
                d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        # distance from small to big
        vals, ix = torch.sort(d.reshape(-1), descending=False)
        ix = ix[vals<=thresh]
        ix = ix.tolist()

        while len(ix) > 0:
            k = ix.pop(0)
            di, dj = k // jlen, k % jlen

            if d[di, dj].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i, j = ii[k], jj[k]
            # bidirectional
            es += [(i, j), ]
            es += [(j, i), ]

            d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        ii, jj = torch.tensor(es, device=self.device).unbind(dim=-1)

        self.add_factors(ii, jj, remove)
