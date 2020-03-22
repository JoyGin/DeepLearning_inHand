import torch

x = torch.tensor(0.5, dtype = torch.float32)
x.required_grad = True
for i in range(30):
    x0 = x.clone()
    x = torch.exp(-x)
    print("%d"%i,"\n %f" %x)
    print("%f"%abs(x0-x))
    print()
    if(torch.abs(x0 - x) < 0.5e-3):break