import torch


# fx = x**2-2*x-torch.exp(x)+2
# fx.backward()
# fx.requires_grad = True
# print(x.grad)

x = torch.tensor([1.0], dtype = torch.float64)
x.requires_grad = True
for i in range(1000):
    fx = x**2-2*x-torch.exp(x)+2
    x0 = x.clone()
    # print(x0.data)
    fx.backward()
    x.data -= fx / x.grad
    x.grad.data.zero_()
    print("%9f"% x)
    print(abs(x0.data - x.data))
    if(torch.abs(x0.data - x.data) < 0.5e-5):break
