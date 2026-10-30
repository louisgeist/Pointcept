import torch
from torch.autograd import Function

from pointops._C import knn_query_cuda


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, offset, new_xyz=None, new_offset=None):
        """
        input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample) -1 is placeholder, dist2: (m, nsample)
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), dtype=torch.int, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float, device=xyz.device)
        knn_query_cuda(
            m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2
        )
        return idx, torch.sqrt(dist2)


knn_query = KNNQuery.apply
