import torch
import torch.nn as nn


class UIBLoss(nn.Module):
    def __init__(self, alpha=4, reduction="sum"):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, model_output, **kwargs):
        pred, user_bound, reg_loss = model_output
        # BPR loss
        # loss = -torch.log(torch.sigmoid(pred[:, :1] - pred[:, 1:]))
        loss_p = -torch.log(torch.sigmoid(pred[:, :1] - user_bound))
        loss_n = -torch.log(torch.sigmoid(user_bound - pred[:, 1:]))
        loss = loss_p + self.alpha * loss_n
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        return loss + reg_loss


_is_hit_cache = {}


def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]["id"] == cacheid:
        return _is_hit_cache[topk]["is_hit"]
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long
        ).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {"id": cacheid, "is_hit": is_hit}
        return is_hit


class _Metric:
    """
    base class of metrics like Recall@k NDCG@k MRR@k
    """

    def __init__(self):
        self.start()

    @property
    def metric(self):
        return self._metric

    @property
    def sum(self):
        return self._sum

    @property
    def cnt(self):
        return self._cnt

    def __call__(self, scores, ground_truth):
        """
        - scores: model output
        - ground_truth: one-hot test dataset shape=(users, all_bundles/all_items).
        """
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        """
        clear all
        """
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum / self._cnt


class Recall(_Metric):
    """
    Recall in top-k samples
    """

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epison = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit / (num_pos + self.epison)).sum().item()


class NDCG(_Metric):
    """
    NDCG in top-k samples
    In this work, NDCG = log(2)/log(1+hit_positions)
    """

    def DCG(self, hit, device=torch.device("cpu")):
        hit = hit / torch.log2(
            torch.arange(2, self.topk + 2, device=device, dtype=torch.float)
        )
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1  # avoid 0/0
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg / idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()


class data_prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_user, self.next_bundle = next(self.loader)
        except StopIteration:
            self.next_user = None
            self.next_bundle = None
            return
        with torch.cuda.stream(self.stream):
            self.next_user = self.next_user.to(self.device, non_blocking=True)
            self.next_bundle = self.next_bundle.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        user = self.next_user
        bundle = self.next_bundle
        self.preload()
        return user, bundle

