import numpy as np
import torch
import torch.nn as nn

text = "some text for testing something in text model"
vocab = list(set(text.lower().split())) #list of uniq words in text



lstm = nn.LSTM(10,16, batch_first=True)

x = torch.randn(1, 5, 10)
y,(h, c) = lstm(x)

print("y", y.size())
print("h", h.size())
print("c",c.size())

x = torch.randn(2, 6, 10)
y,(h, c) = lstm(x)

print("y2", y.size())
print("h2", h.size())
print("c2",c.size())