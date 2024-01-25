import torch
from get_batch import get_pair_batch
import GLOBAL

batch_h, batch_r, batch_t, length = get_pair_batch(0, [1,2,3], 3, 4)
print(batch_h.shape, batch_r.shape, batch_t.shape)