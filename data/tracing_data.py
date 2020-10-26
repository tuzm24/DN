import os
from data import srdata
import glob
from help_func.CompArea import PictureFormat
from data import common
import imageio
import numpy as np
import random
from help_func.CompArea import TuList
from help_func.CompArea import TuDataIndex
from help_func.help_python import myUtil
from help_func.typedef import *




class tracing_data(srdata.SRData):
    ppsfilename = 'PPSParam.npz'
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.end -=self.begin
        self.begin = 1
        super(tracing_data, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.pps_data  = self._scanPPSData()
        self.blocks_ext = '.npy'
        self.blocks = {}
        self._scanblockData()
        self.idxes = [TuDataIndex.NAME_DIC[x] for x in self.args.tu_data_type]
        self.__getitem__(0)

    def _scanPPSData(self):
        named_ppsFile = []
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            named_ppsFile.append(os.path.join(
                self.dir_lr, '{}/{}/{}'.format(
                    'BLOCK', filename, self.ppsfilename
                )
            ))
        return named_ppsFile


    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.data_types]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.data_types):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr

    def _scanblockData(self):
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            block_dir = os.path.join(self.dir_lr,
                                     'BLOCK', filename)
            for file in myUtil.getFileList(block_dir, self.blocks_ext):
                key, _ = os.path.splitext(os.path.basename(file))
                if key in self.blocks:
                    self.blocks[key].append(file)
                else:
                    self.blocks[key] = [file]

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL])
        self.dir_lr = os.path.join(self.apath)
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.npz', '.npz')


    @staticmethod
    def read_npz_file(f):
        def UpSamplingChroma(UVPic):
            return UVPic.repeat(2, axis=0).repeat(2, axis=1)
        f = np.load(f)
        return np.stack((f['Y'], UpSamplingChroma(f['Cb']), UpSamplingChroma(f['Cr'])), axis=2)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('npz') >= 0:
            hr = self.read_npz_file(f_hr)
            lr = []
            for flr in self.images_lr:
                lr.append(self.read_npz_file(flr[idx]))
            # lr = self.read_npz_file(f_lr)
        else:
            assert 0

        return lr, hr, filename

    def get_patch(self, lr, hr):
        def _get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
            ih, iw = args[0].shape[:2]

            if not input_large:
                p = scale if multi else 1
                tp = p * patch_size
                ip = tp // scale
            else:
                tp = patch_size
                ip = patch_size

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)

            if not input_large:
                tx, ty = scale * ix, scale * iy
            else:
                tx, ty = ix, iy

            ret = [
                args[0][ty:ty + tp, tx:tx + tp, :],
                [a[iy:iy + ip, ix:ix + ip, :] for a in args[1]]
            ]

            return ret[0], ret[1], (ty, tp, tx, tp)

        scale = self.scale[self.idx_scale]
        tpy, tpx = hr.shape[:2]
        imgy, imgx = hr.shape[:2]
        ty, tx = 0, 0
        if self.train:
            hr, lr, (ty, tpy, tx, tpx) = _get_patch(
                hr, lr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=True
            )
            if self.args.no_augment: lr, hr = common.augment(lr, hr)

        return lr, hr, (ty, tpy, tx, tpx), (imgy, imgx)
    # 0:xpos, 1:ypos, 2:width, 3:height
    def getBlock2d(self, y, dy, x, dx, idx, name, value_idx = 0):
        block2d = np.zeros((dy, dx))
        dy = y + dy
        dx = x + dx
        block = np.load(self.blocks[name][idx])
        filtered = block[:, ~np.any([(block[3] + block[1]) < y,
                                     (block[0] + block[2]) < x,
                                     block[1]>dy,
                                     block[0]>dx
                                     ], axis=0)]
        filtered[0, filtered[0]<x] = x
        filtered[1, filtered[1]<y] = y
        filtered[2, (filtered[0] + filtered[2]) > dx] = dx - filtered[0, (filtered[0]+filtered[2] > dx)]
        filtered[3, (filtered[1] + filtered[3]) > dy] = dy - filtered[1, (filtered[1] + filtered[3]) > dy]
        filtered[0,:] -= x
        filtered[1,:] -= y
        for _xp, _yp, _w, _h, *values in filtered.T:
            block2d[_yp:_yp+_h, _xp:_xp+_w] = values[value_idx]
        return block2d

    def getBlockScalar(self, idx, name, value_idx=0):
        block = np.load(self.blocks[name][idx])
        return block.mean(axis=0)

    def getTuMask(self, ty, tpy, tx, tpx, h, w, filename):
        tulist = TuList(np.load(
            os.path.join(self.dir_lr, '{}/{}{}'.format(
                self.args.tu_data, os.path.splitext(os.path.basename(filename))[0], self.ext[1]
            ))
        )['LUMA'])
        return TuList.NormalizedbyMinMaxAll(tulist.getTuMaskFromIndexes(self.idxes, h, w)[:, ty:ty+tpy, tx:tx+tpx], self.idxes).astype('float32')


    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr, pos, imgshape = self.get_patch(lr, hr)
        self.getBlock2d(*pos, idx, BlockType.Luma_IntraMode)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return common.np2Tensor2(lr), common.np2Tensor2([hr])[0]