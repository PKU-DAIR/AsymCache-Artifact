import torch
ret_0 = torch.load('ret_0.pt', map_location=torch.device('cpu')) # pod
ret_1 = torch.load('ret_1.pt', map_location=torch.device('cpu')) # flash

assert ret_0.shape == ret_1.shape

len = ret_0.shape[0]
print([torch.allclose(ret_0[i], ret_1[i], 1e-3, 1e-3) for i in range(len)])

for i in range(0, len):
    print(f'idx={i}', end='')
    print([torch.allclose(ret_0[i][j], ret_1[i][j], 1e-3, 1e-3) for j in range(32)])