import torch
a = torch.tensor([[2, 2],[3,3]],dtype=float) # 缺失情况下默认 requires_grad = False
a.requires_grad_(True)
b = (a * 3)
print(a.requires_grad) # True
c = (b * b).mean()
print(c)
c.backward()
print(a.grad)




