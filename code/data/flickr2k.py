import os
from data import srdata

class FLICKR2K(srdata.SRData):
    def __init__(self, args, name='Flickr2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[2]
        else:
            if args.test_only and len(data_range) == 2:
                data_range = data_range[1]
            else:
                data_range = data_range[3]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(FLICKR2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(FLICKR2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(FLICKR2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Flickr2K_HR')
        self.dir_lr = os.path.join(self.apath, 'Flickr2K_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
