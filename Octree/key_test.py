import torch
r512=torch.arange(2**3,dtype=torch.int64)
depth=3


key=r512
x=torch.zeros_like(key)
y=torch.zeros_like(key)
z=torch.zeros_like(key)


i=0

x=x|((key&(1<<(3*i+2)))>>(2*i+2))
y=y|((key&(1<<(3*i+1)))>>(2*i+1))
z=z|((key&(1<<(3*i+0)))>>(2*i+0))
print(bin(x))
print(bin(y))
print(bin(z))












