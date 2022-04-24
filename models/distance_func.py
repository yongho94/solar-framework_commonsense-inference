import torch


class NonLinearHeadDistance(torch.nn.Module):
    def __init__(self):
        super(NonLinearHeadDistance, self).__init__()

    def forward(self, batch_s, batch_ro):
        l2_s = batch_s / torch.linalg.norm(batch_s, dim=-1, ord=2, keepdim=True)
        l2_ro = batch_ro / torch.linalg.norm(batch_ro, dim=-1, ord=2, keepdim=True)

        l2_ro = l2_ro.transpose(0, 1)

        final_dist = torch.matmul(l2_s, l2_ro)
        return final_dist


class MultiHeadDistance(torch.nn.Module):
    def __init__(self, head_option):
        super(MultiHeadDistance, self).__init__()
        self.head_num = head_option['head_num']
        self.head_dim = head_option['head_dim']
        self.pool = None
        if head_option['pool_method'] == 'maxpool':
            self.max_pool = torch.nn.MaxPool1d(self.head_num)
        else:
            raise NotImplementedError

    def forward(self, batch_s, batch_ro):
        batch_s = batch_s.view(list(batch_s.shape[:-1]) + [self.head_num, self.head_dim])
        batch_ro = batch_ro.view(list(batch_ro.shape[:-1]) + [self.head_num, self.head_dim])

        l2_s = batch_s / torch.linalg.norm(batch_s, dim=-1, ord=2, keepdim=True)
        l2_ro = batch_ro / torch.linalg.norm(batch_ro, dim=-1, ord=2, keepdim=True)

        l2_s = l2_s.permute([1, 0, 2])
        l2_ro = l2_ro.permute([1, 0, 2])
        l2_ro = l2_ro.transpose(1, 2)

        dist = torch.matmul(l2_s, l2_ro)
        dist = dist.permute([1, 2, 0])

        final_dist = self.max_pool(dist).squeeze(-1)
        return final_dist

