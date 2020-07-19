import os
import imageio
import torch

# batch_size = 3
# batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
#
# data_dir = \
#     '/home/sanyog.vyawahare/Desktop/Study/Learn PyTorch/dlwpt-code-master/data/p1ch4/image-cats/'
# filenames = [name for name in os.listdir(data_dir)
#              if os.path.splitext(name)[-1] == '.png']
# for i, filename in enumerate(filenames):
#     img_arr = imageio.imread(os.path.join(data_dir, filename))
#     img_t = torch.from_numpy(img_arr)
#     img_t = img_t.permute(2, 0, 1)
#     img_t = img_t[:3]
#     batch[i] = img_t
# print(batch[1], batch.shape)
# n_channels = batch.shape[1]
#
# batch = batch.float()
# batch /= 255.0
# for c in range(n_channels):
#     mean = torch.mean(batch[:, c])
#     std = torch.std(batch[:, c])
#     print(batch[:, c].shape)
#     batch[:, c] = (batch[:, c] - mean) / std

# dir_path = "/home/sanyog.vyawahare/Desktop/Study/Learn PyTorch/dlwpt-code-master/data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
# load_dir = imageio.volread(dir_path, "DICOM")
#
# print(load_dir.shape)
#
# tensor = torch.from_numpy(load_dir).float()
# tensor = tensor.unsqueeze(0)
# print(tensor.shape)

