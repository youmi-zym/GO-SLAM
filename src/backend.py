import torch
import numpy as np
from copy import deepcopy
from .factor_graph import FactorGraph


class Backend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.device = args.device
        self.update_op = net.update

        self.upsample = cfg['tracking']['upsample']
        self.beta = cfg['tracking']['beta']
        self.backend_thresh = cfg['tracking']['backend']['thresh']
        self.backend_radius = cfg['tracking']['backend']['radius']
        self.backend_nms = cfg['tracking']['backend']['nms']

        self.backend_loop_window = cfg['tracking']['backend']['loop_window']
        self.backend_loop_thresh = cfg['tracking']['backend']['loop_thresh']
        self.backend_loop_radius = cfg['tracking']['backend']['loop_radius']
        self.backend_loop_nms = cfg['tracking']['backend']['loop_nms']

    @torch.no_grad()
    def ba(self, t_start, t_end, steps, graph, nms, radius, thresh, max_factors, t_start_loop=None, loop=False, motion_only=False):
        """ main update """
        if t_start_loop is None or not loop:
            t_start_loop = t_start
        assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'

        ilen = (t_end - t_start_loop)
        jlen = (t_end - t_start)
        ix = torch.arange(t_start_loop, t_end)
        jx = torch.arange(t_start, t_end)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=self.beta)
        rawd = deepcopy(d).reshape(ilen, jlen)
        d[ii - radius < jj] = np.inf
        d[d > thresh] = np.inf
        d = d.reshape(ilen, jlen)

        es = []
        # build edges within local window [i-rad, i]
        for i in range(t_start_loop, t_end):
            if self.video.stereo and not loop:
                es.append((i, i))
                di, dj = i-t_start_loop, i-t_start
                d[di, dj] = np.inf

            for j in range(max(i-radius, t_start_loop), i):  # j in [i-radius, i-1]
                es.append((i, j))
                es.append((j, i))
                di, dj = i-t_start_loop, j-t_start
                d[di, dj] = np.inf
                d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        # distance from small to big
        vals, ix = torch.sort(d.reshape(-1), descending=False)
        ix = ix[vals<=thresh]
        ix = ix.tolist()

        n_neighboring = 1
        while len(ix) > 0:
            k = ix.pop(0)
            di, dj = k // jlen, k % jlen

            if d[di, dj].item() > thresh:
                continue

            if len(es) > max_factors:
                break

            i, j = ii[k], jj[k]
            # bidirectional
            if loop:
                sub_es = []
                num_loop = 0
                for si in range(max(i-n_neighboring, t_start_loop), min(i+n_neighboring+1, t_end)):
                    for sj in range(max(j-n_neighboring, t_start), min(j+n_neighboring+1, t_end)):
                        if rawd[(si-t_start_loop), (sj-t_start)] <= thresh:
                            num_loop += 1
                            if si != sj:
                                sub_es += [(si, sj)]
                if num_loop > int(((n_neighboring * 2 + 1) ** 2) * 0.5):
                    es += sub_es
            else:
                es += [(i, j), ]
                es += [(j, i), ]

            d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        if len(es) < 3:
            return 0

        ii, jj = torch.tensor(es, device=self.device).unbind(dim=-1)

        graph.add_factors(ii, jj, remove=True)

        edge_num = len(graph.ii)

        graph.update_lowmem(
            t0=t_start_loop+1,  # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
            t1=t_end,
            iters=2,
            use_inactive=False,
            steps=steps,
            max_t=t_end,
            ba_type='dense',
            motion_only=motion_only,
        )


        graph.clear_edges()

        torch.cuda.empty_cache()

        self.video.dirty[t_start:t_end] = True

        return edge_num

    @torch.no_grad()
    def dense_ba(self, t_start, t_end, steps=6, motion_only=False):
        nms = self.backend_nms
        radius = self.backend_radius
        thresh = self.backend_thresh
        n = t_end - t_start
        max_factors = (int(self.video.stereo) + (radius + 2) * 2) * n

        graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt', max_factors=max_factors, upsample=self.upsample)
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, thresh, max_factors, motion_only=motion_only)

        del graph

        return n, n_edges


    @torch.no_grad()
    def loop_ba(self, t_start, t_end, steps=6, motion_only=False, local_graph=None):
        radius = self.backend_radius
        window = self.backend_loop_window
        max_factors = 8 * window
        nms = self.backend_loop_nms
        thresh = self.backend_loop_thresh
        t_start_loop = max(0, t_end - window)

        graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt', max_factors=max_factors, upsample=self.upsample)
        if local_graph is not None:
            copy_attr = ['ii', 'jj', 'age', 'net', 'target', 'weight']
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val))

        left_factors = max_factors - len(graph.ii)
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, thresh, left_factors, t_start_loop=t_start_loop, loop=True, motion_only=motion_only)

        del graph

        return t_end - t_start_loop, n_edges
