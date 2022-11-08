import torch
import torch.nn as nn

in_features = 4
out_features = 4

N = 6

W = nn.Parameter(torch.empty(size=(in_features, out_features))) # torch.Size([4, 4])
a = nn.Parameter(torch.randn(size=(2*out_features, 1)))         # torch.Size([8, 1])

h = nn.Parameter(torch.empty(size=(N, in_features)))            # torch.Size([6, 4])

# print(W)
print(a)
# print(h)

Wh = torch.mm(h, W)                                             # torch.Size([6, 4])
# print(Wh.shape)

# print(a[:out_features, :])

Wh1 = torch.matmul(Wh, a[:out_features, :])                     # torch.Size([6, 1])
Wh2 = torch.matmul(Wh, a[out_features:, :])
# print(Wh1)
# print(Wh2.T)
print(a[:out_features, :])
print(a[out_features:, :])

e = Wh1 + Wh2.T                                                 # # torch.Size([6, 6])
# print(e)


aa = nn.Parameter(torch.empty(size=(3, 1)))
bb = nn.Parameter(torch.empty(size=(3, 1)))
# print(aa)
# print(bb.T)
ee = aa + bb.T
# print(ee)