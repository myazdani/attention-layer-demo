import torch, math, sys

from torch import nn
from torch.nn import functional as F


class DataUtil:
    def __init__(self, seq_height_min, seq_height_max, seq_width_min, 
                 seq_width_max, seq_length, group_by_locations):
        self.seq_height_min, self.seq_height_max = seq_height_min, seq_height_max
        self.seq_width_min, self.seq_width_max = seq_width_min, seq_width_max
        self.seq_length = seq_length
        self.group_by_locations = group_by_locations

    def positions_to_sequences(self, tr = None, bx = None, noise_level = 0.3):
        st = torch.arange(self.seq_length).float()
        st = st[None, :, None]
        tr = tr[:, None, :, :]
        bx = bx[:, None, :, :]

        xtr =            torch.relu(tr[..., 1] - torch.relu(torch.abs(st - tr[..., 0]) - 0.5) * 2 * tr[..., 1] / tr[..., 2])
        xbx = torch.sign(torch.relu(bx[..., 1] - torch.abs((st - bx[..., 0]) * 2 * bx[..., 1] / bx[..., 2]))) * bx[..., 1]

        x = torch.cat((xtr, xbx), 2)

        # u = x.sign()
        u = F.max_pool1d(x.sign().permute(0, 2, 1), kernel_size = 2, stride = 1).permute(0, 2, 1)

        collisions = (u.sum(2) > 1).max(1).values
        y = x.max(2).values

        return y + torch.rand_like(y) * noise_level - noise_level / 2, collisions

    def generate_sequences(self, nb):

        # Position / height / width

        tr = torch.empty(nb, 2, 3)
        tr[:, :, 0].uniform_(self.seq_width_max/2, self.seq_length - self.seq_width_max/2)
        tr[:, :, 1].uniform_(self.seq_height_min, self.seq_height_max)
        tr[:, :, 2].uniform_(self.seq_width_min, self.seq_width_max)

        bx = torch.empty(nb, 2, 3)
        bx[:, :, 0].uniform_(self.seq_width_max/2, self.seq_length - self.seq_width_max/2)
        bx[:, :, 1].uniform_(self.seq_height_min, self.seq_height_max)
        bx[:, :, 2].uniform_(self.seq_width_min, self.seq_width_max)
        
        if self.group_by_locations:
            a = torch.cat((tr, bx), 1)
            v = a[:, :, 0].sort(1).values[:, 2:3]
            mask_left = (a[:, :, 0] < v).float()
            h_left = (a[:, :, 1] * mask_left).sum(1) / 2
            h_right = (a[:, :, 1] * (1 - mask_left)).sum(1) / 2
            valid = (h_left - h_right).abs() > 4
        else:
            valid = (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4) & (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4)

        input, collisions = self.positions_to_sequences(tr, bx)

        if self.group_by_locations:
            a = torch.cat((tr, bx), 1)
            v = a[:, :, 0].sort(1).values[:, 2:3]
            mask_left = (a[:, :, 0] < v).float()
            h_left = (a[:, :, 1] * mask_left).sum(1, keepdim = True) / 2
            h_right = (a[:, :, 1] * (1 - mask_left)).sum(1, keepdim = True) / 2
            a[:, :, 1] = mask_left * h_left + (1 - mask_left) * h_right
            tr, bx = a.split(2, 1)
        else:
            tr[:, :, 1:2] = tr[:, :, 1:2].mean(1, keepdim = True)
            bx[:, :, 1:2] = bx[:, :, 1:2].mean(1, keepdim = True)

        targets, _ = self.positions_to_sequences(tr, bx)

        valid = valid & ~collisions
        tr = tr[valid]
        bx = bx[valid]
        input = input[valid][:, None, :]
        targets = targets[valid][:, None, :]

        if input.size(0) < nb:
            input2, targets2, tr2, bx2 = self.generate_sequences(nb - input.size(0))
            input = torch.cat((input, input2), 0)
            targets = torch.cat((targets, targets2), 0)
            tr = torch.cat((tr, tr2), 0)
            bx = torch.cat((bx, bx2), 0)

        return input, targets, tr, bx