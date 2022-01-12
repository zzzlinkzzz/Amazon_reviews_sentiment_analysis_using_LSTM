from torch.utils.data import Dataset, DataLoader

from torch import tensor
import numpy as np
from linecache import getline

class MyDataset(Dataset):
    def __init__(self, file_path, text_seg):
        self._file_path = file_path
        self._total_data = 0
        self._text_seg = text_seg
        self._max_length = 200

        with open(file_path, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        #read line
        line = getline(self._file_path, idx + 1)
        line = eval(line)
        text = line[1:][:self._max_length]
        
        # embedding
        text_mat = np.empty((300,0), float)
        for t in text:
            v = getline('.vector_cache/glove.840B.300d.txt',t+1).split(' ')[1:]
            v = np.array(list(map(float,v)))
            text_mat = np.c_[text_mat,v]
        
        # padding
        if text_mat.shape[1] < self._max_length:
            text_mat = np.c_[np.zeros((300,(self._max_length - text_mat.shape[1]))),text_mat]
        
        return {
                'label': tensor((int(line[0])-1)).long(),
                'text' : tensor(text_mat.T).float()
                }

    def __len__(self):
        return self._total_data
    
def make_dataloaders(path, data_dir, text_seg, batch_size, num_workers):
    train_dataset = MyDataset(f'{path}/{data_dir}/v_train.txt', text_seg)
    test_dataset = MyDataset(f'{path}/{data_dir}/v_test.txt', text_seg)
        
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=False)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False)
    return train_loader, test_loader

# unit test
if __name__ == "__main__":
    batch_size = 2000
    num_workers = 2
    train_loader,test_loader = make_dataloaders('./dataset', 'sample', 'both', batch_size, num_workers)

    count = 0
    for i in train_loader:
        if count == 1:
            break
        print(i['text'])
        print(i['label'])
        count+=1