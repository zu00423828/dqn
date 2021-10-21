from torch.utils.data import Dataset
import numpy as np
class ENV(Dataset):
    def __init__(self,n_actions,n_states,data_len=200000):
        self.n_actions=n_actions
        self.n_states=n_states
        self.data_len=data_len
    def __len__(self):
        return self.data_len
    def __getitem__(self, index):
        origin=np.random.rand(self.n_states)
        half_index=(self.n_states//2)
        x=origin.copy()
        y=np.zeros((4))
        label=np.argmin(abs(np.delete(origin,half_index)-origin[half_index]),axis=0)
        y[label]=1.
        return x.astype(np.float32),y.astype(np.float32)
