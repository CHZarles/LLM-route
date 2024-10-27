# Q: 用同目录下的train文件夹实现一个数据集


# 1. 引入Dataset类
from torch.utils.data import  Dataset
import os

# 2. 继承 class Dataset
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        """
        初始化数据
        :param root_dir: 数据的根目录，ie ./train
        :param label_dir: 在./train里面，同类的数据都被放在同一个文件夹了 , ie ./train/ant
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 拼接成实际文件路径
        file_paths = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(file_paths)


    # 3.overwrite __getitem__(self, index)
    #   overwrite __len__()
    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.label_dir, self.image_list[index])
        return image_name , self.label_dir

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    root_dir = os.path.join("train")
    a = MyData(root_dir, "ants")
    b = MyData(root_dir, "bees")
    c = a + b
    assert b[0] == c[len(a)]
    print(b[2])
