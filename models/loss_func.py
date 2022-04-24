import torch
EPS = 0.0001


class NT_Logistic:
    def __init__(self, temp=0.1):
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()

    def get_loss(self, dist, pos_sample):
        neg_sample = torch.ones_like(pos_sample) - pos_sample
        pos_dist = torch.mul(dist, pos_sample)
        neg_dist = torch.mul(dist, neg_sample)
        avg = (pos_sample / torch.sum(pos_sample, -1)) + (neg_sample / torch.sum(neg_sample, -1))
        logged_dist = torch.log(self.sigmoid((pos_dist - neg_dist) / self.temp) + 0.0001)

        pos_dists = torch.sum(pos_dist) / torch.sum(pos_sample)
        neg_dists = torch.sum(neg_dist) / torch.sum(neg_sample)
        avg_dist = torch.sum(torch.mul(logged_dist, avg), -1)
        avg_dist = -torch.mean(avg_dist)
        return avg_dist, pos_dists, neg_dists


'''
class NT_Logistic:
    def __init__(self, temp=0.1):
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()

    def get_loss(self, dist, pos_sample):
        pos_sample = pos_sample
        neg_sample = torch.ones_like(pos_sample) - pos_sample

        pos_dist, pos_num = self.get_each_loss(dist, pos_sample)
        neg_dist, neg_num = self.get_each_loss(dist, neg_sample)
        logged_dist = torch.log(self.sigmoid((pos_dist - neg_dist) / self.temp) + 0.0001)

        avg = torch.zeros_like(dist)
        if pos_num.item() != 0:
            avg += pos_sample / pos_num
        if neg_num.item() != 0:
            avg += neg_sample / neg_num

        loss = -torch.sum(logged_dist * avg)
        print(loss)
        print(pos_num, neg_num)
        print(logged_dist[0][:50])
        #print(avg[0])
        pos_dist_total = torch.sum(pos_dist)
        neg_dist_total = torch.sum(neg_dist)

        pos_dists = (pos_dist_total.item(), pos_num.item())
        neg_dists = (neg_dist_total.item(), neg_num.item())

        return loss, pos_dists, neg_dists




        pos_dist = torch.mul(dist, pos_sample) + EPS
        neg_dist = torch.mul(dist, neg_sample) + EPS
        avg = (pos_sample / (torch.sum(pos_sample, -1) + EPS)) + (neg_sample / (torch.sum(neg_sample, -1) + EPS))
        logged_dist = torch.log(self.sigmoid((pos_dist - neg_dist) / self.temp) + 0.0001)

        pos_dists = torch.sum(pos_dist) / torch.sum(pos_sample) + EPS
        neg_dists = torch.sum(neg_dist) / torch.sum(neg_sample) + EPS
        avg_dist = torch.sum(torch.mul(logged_dist, avg), -1)
        avg_dist = -torch.mean(avg_dist)
        return avg_dist, pos_dists, neg_dists

    def get_each_loss(self, dist, sample_mask):
        sample_num = torch.sum(sample_mask)
        sample_dist = torch.mul(dist, sample_mask)
        return sample_dist, sample_num
'''