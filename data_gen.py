from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class Linear_Data(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        _min = 0
        _max = 1000
        m = 1
        x = np.random.uniform(_min, _max)
        return {
                "input": np.array([(x +  np.random.normal(0, 1)) / 1000], dtype=np.float32),
                "output": np.array([(m * x) / 1000], dtype=np.float32)
            }


if __name__ == "__main__":
    x, y = [], []
    linear_data = Linear_Data()
    for i in range(100):
        d = linear_data.__getitem__(i)
        x.append(d["input"])
        y.append(d["output"])
    plt.plot(x, y, "ob")
    plt.plot(x, y)
    plt.show()