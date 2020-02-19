import torch
import numpy as np
import math
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape
        # print(_, z, h, w)
        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        # print(nzhw)
        pad = [ [0, 0],
                [math.ceil(margin), math.ceil(nz * side_len - z + margin)],
                [math.ceil(margin), math.ceil(nh * side_len - h + margin)],
                [math.ceil(margin), math.ceil(nw * side_len - w + margin)]]
        # print(pad)
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = math.ceil(iz * side_len)
                    ez = math.ceil((iz + 1) * side_len + 2 * margin)
                    sh = math.ceil(ih * side_len)
                    eh = math.ceil((ih + 1) * side_len + 2 * margin)
                    sw = math.ceil(iw * side_len)
                    ew = math.ceil((iw + 1) * side_len + 2 * margin)

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])
        output = -1000000 * np.ones((
            int(nz * side_len),
            int(nh * side_len),
            int(nw * side_len),
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = math.ceil(iz * side_len)
                    ez = math.ceil((iz + 1) * side_len)
                    sh = math.ceil(ih * side_len)
                    eh = math.ceil((ih + 1) * side_len)
                    sw = math.ceil(iw * side_len)
                    ew = math.ceil((iw + 1) * side_len)
                    margin = int(margin)
                    side_len = int(side_len)
                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output 
