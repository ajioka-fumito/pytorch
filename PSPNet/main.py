from md.loader import Mydataset,Make_dataset_paths
from torch.utils.data import DataLoader
from torchvision import transforms
def main(parameter):
    all_paths = Make_dataset_paths(parameter["data_dir"],parameter["test_rate"])
    train_paths,test_paths = all_paths.dataset_to_train_test()
    train_dataset = Mydataset(train_paths[0],train_paths[1],transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset)


    
if __name__ == "__main__":
    parameter = {"data_dir":"./data/","test_rate":0.1}
    main(parameter)