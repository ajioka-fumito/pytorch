import torch
from torch.utils.data import DataLoader,Dataset
from md.loader import Make_dataset_paths,MyDataset
from torchvision import transforms
parameter = {"test_image_dir":"./data/test/image",
             "test_label_dir":"./data/test/label",}

model = torch.load("./model").to("cuda").float()

paths = Make_dataset_paths(parameter)

test_image_paths,test_label_paths = paths.generate_test_paths()

dataset = MyDataset(test_image_paths,test_label_paths,transform=transforms.Compose([transforms.ToTensor()]))

ld = DataLoader(dataset,batch_size=1)

for i,(image,_) in enumerate(ld):
    image = image.to("cuda").float()
    print(image.shape)
    predict = model(image)
    print(predict.shape)
    break

predict = predict.squeeze()
print(predict.shape)
import numpy as np

predict = predict.cpu().detach().numpy()
predict = np.transpose(predict,(1,2,0))
print(np.shape(predict))
predict = np.argmax(predict,axis=2)

from PIL import Image

