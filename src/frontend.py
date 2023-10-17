import torch
from copy import deepcopy
from time import gmtime, strftime, time

from .factor_graph import FactorGraph
from .backend import Backend as LoopClosing


class Frontend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.update_op = net.update
        self.warmup = cfg['tracking']['warmup']
        self.upsample = cfg['tracking']['upsample']
        self.beta = cfg['tracking']['beta']
        self.verbose = cfg['verbose']

        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors']
        self.frontend_nms = cfg['tracking']['frontend']['nms']
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh']
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.frontend_thresh = cfg['tracking']['frontend']['thresh']
        self.frontend_radius = cfg['tracking']['frontend']['radius']
        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, args, cfg)
        self.last_loop_t = -1

        self.graph = FactorGraph(
            video, net.update,
            device=args.device,
            corr_impl='volume',
            max_factors=self.frontend_max_factors,
            upsample=self.upsample
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontend variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # build edges between [t1-5, video.counter] and [t1-window, video.counter]
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
                                         rad=self.frontend_radius, nms=self.frontend_nms,
                                         thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
                                                  self.video.disps_sens[self.t1-1],
                                                  self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(t0=None, t1=None, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1-2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        else:
            cur_t = self.video.counter.value
            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Loop BA'
            msg = f'\n\n {now} : [{t_start}, {cur_t}]; Current Keyframe is {cur_t}, last is {self.last_loop_t}.'
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, motion_only=False, local_graph=self.graph)

                print(msg + f' {n_kf} KFs, last KF is {self.last_loop_t}! \n')
                self.last_loop_t = cur_t

            else:
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True)

        # set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        # build edges between nearby(radius <= 3) frames within local windown [t0, t1]
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        # build edges between [t0, video.counter] and [t1, video.counter]
        self.graph.add_proximity_factors(t0=0, t1=0, rad=2, nms=2,
                                         thresh=self.frontend_thresh,
                                         remove=False)

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()


        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        else:
            pass




