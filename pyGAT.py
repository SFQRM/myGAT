# 加载数据集用到的依赖
from process_data import load_data

# 构建模型用到的依赖
import torch.nn.functional as F
from model import GAT

# 构建优化器用到的依赖
import torch.optim as optim

# 训练用到的依赖
from torch.autograd import Variable
import time

# 引用一些工具类函数
from utils import accuracy, clustering_coefficient


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    """
        features: (2708, 1433)
        adj: (2708, 2708)
    """
    output = model(features, adj, cc)       # torch.Size([2708, 7])
    # print(output[idx_train])            # torch.Size([140, 7])
    # print(labels[idx_train])            # torch.Size([140])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    """
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    """

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # return loss_val.data.item()


def compute_test(model, cc):
    model.eval()
    output = model(features, adj, cc)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':
    # 加载数据
    path = "./data/cora/"
    labels, features, adj, idx_train, idx_val, idx_test = load_data(path=path)

    cc = clustering_coefficient(adj)
    # print(cc)

    # 构建模型
    model = GAT(num_features=features.shape[1],
                num_hidden=8,
                num_class=int(labels.max()) + 1,
                dropout=0.6,
                num_heads=8,
                alpha=0.2)

    # 构建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 训练准备，加载cuda
    cuda = True
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    cc = cc.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    features, adj, labels, cc = Variable(features), Variable(adj), Variable(labels), Variable(cc)

    # 训练模型
    t_total = time.time()
    epochs = 1000
    for epoch in range(epochs):
        train(epoch)

    # for name, param in model.named_parameters():
    #     print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # 测试模型
    compute_test(model=model, cc=cc)