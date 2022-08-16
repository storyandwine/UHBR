import argparse
from time import time

import torch
from torch.utils.data import DataLoader
import dataset
from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Go UHBR for bundle recommendation")
    parser.add_argument("--lr", type=float, default=5e-3, help="the learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Youshu",
        help="available datasets: [Youshu, NetEase]",
    )
    parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")
    parser.add_argument("--dp", type=float, default=0.2, help="the dropout rate")
    parser.add_argument("--alpha", type=int, default=8, help="alpha in UIBloss")
    parser.add_argument("--l2_norm", type=float, default=0.1, help="l2 norm")
    return parser.parse_args()


def train(model, epoch, loader, optim, device, loss_func):
    prefetcher = data_prefetcher(loader, device)
    model.train()
    start = time()
    i = 0
    users, bundles = prefetcher.next()
    while users is not None:
        i += 1
        optim.zero_grad()
        modelout = model(users, bundles)
        loss = loss_func(modelout, batch_size=loader.batch_size)
        loss.backward()
        optim.step()
        if i % 20 == 0:
            print(
                "U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    (i + 1) * loader.batch_size,
                    len(loader.dataset),
                    100.0 * (i + 1) / len(loader),
                    loss,
                )
            )
        users, bundles = prefetcher.next()
    print("Train Epoch: {}: time = {:d}s".format(epoch, int(time() - start)))
    return loss


def test(model, loader, device, metrics):
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate()
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8 * train_mask_u_b.to(device)
            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
    print("Test: time={:.5f}s".format(int(time() - start)))
    for metric in metrics:
        metric.stop()
        print("{}:{}".format(metric.get_title(), metric.metric), end="\t")
    print("")
    return metrics


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    device = torch.device("cuda")
    set_seed(123)
    (
        bundle_train_data,
        bundle_test_data,
        item_data,
        assist_data,
    ) = dataset.get_dataset(args.dataset, path="./data")
    if args.dataset == "Youshu":
        batch_size = 1024
    else:
        batch_size = 2048
    train_loader = DataLoader(
        bundle_train_data, batch_size, True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        bundle_test_data, 4096, False, num_workers=16, pin_memory=True
    )

    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    metrics = [
        Recall(20),
        NDCG(20),
        Recall(40),
        NDCG(40),
        Recall(80),
        NDCG(80),
    ]
    loss_func = UIBLoss(alpha=args.alpha)
    graph = [ui_graph, bi_graph, ub_graph]
    model = UHBR(graph, device, args.dp, args.l2_norm).to(device)
    print("num parameters")
    print(sum(p.numel() for p in model.parameters()))
    # op
    op = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        op, milestones=[35, 55, 75], gamma=0.5
    )
    for epoch in range(args.epochs):

        train(model, epoch + 1, train_loader, op, device, loss_func)
        test(model, test_loader, device, metrics)
        scheduler.step()


if __name__ == "__main__":
    main()

