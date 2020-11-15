from glob import glob
import numpy as np
import cv2

class DataLoader():
    def __init__(self, dataset_name, img_res=(28, 28), mem_load=True, extension="jpg"):

        self.mem_load = mem_load
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.path = glob(f"{self.dataset_name}/{self.img_res[0]}by{self.img_res[1]}/*.{extension}")
        self.n_data = len(self.path)
        if self.mem_load:
            # scale image RGB values from 0,255 to -1,1
            self.total_imgs = np.array(list(map(self.imread, self.path))) / 127.5 -1.

    def load_data(self, batch_size=1, is_testing=False):
        imgs = [] # images to be returned
        if self.mem_load:
            # return random batch from imgs loaded in memory
            idx = np.random.choice(range(self.n_data), size=batch_size)
            for i in idx:
                imgs.append(self.total_imgs[i])
            imgs = np.array(imgs)
        else:
            # read random batch of images
            batch_images = np.random.choice(self.path, size=batch_size)

            for img_path in batch_images:
                img = self.imread(img_path)
                imgs.append(img)

            # scale image RGB values from 0,255 to -1,1
            imgs = np.array(imgs) / 127.5 - 1.
        return imgs

    def get_n_data(self):
        return self.n_data

    def imread(self, path):
        img = cv2.imread(path)
        # cv2 reads image in BGR - convert to RGB to be compatible with matplotlib
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    dl = DataLoader(dataset_name="celeba", img_res=(28,28), mem_load=True)
    print(dl.get_n_data())