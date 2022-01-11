from torch.utils.data import Dataset, DataLoader

from torch import tensor
import numpy as np
import linecache

from torchtext.vocab import GloVe
embedding = GloVe(name = '840B', dim = 300)

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
'with', 'about', 'against', 'between', 'into', 'through', 'during', 
'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
'once', 'here', 'there', 'when', 'where', 'why', 'how']

class MyDataset(Dataset):
    def __init__(self, file_path, text_seg):
        self._file_path = file_path
        self._total_data = 0
        self._text_seg = text_seg

        with open(file_path, "r") as f:
            self._total_data = len(f.readlines())
            
        if self._text_seg == 'title':
            self._max_length = 20
        elif self._text_seg == 'review':
            self._max_length = 200
        else:
            self._max_length = 200

    def __getitem__(self, idx):
        #read line
        line = linecache.getline(self._file_path, idx + 1).replace('\n','').lower()
        title, review = line[11:].split(': ',1)
        
        if self._text_seg == 'title':
            text = title
        elif self._text_seg == 'review':
            text = review
        else:
            text = title + '. ' + review
        
        # tokenize
        text = tokenizer(text)[:self._max_length]
        
        # embedding
        text_mat = np.empty((300,0), float)
        for t in text:
            if t not in stopwords:
                try:
                    text_mat = np.c_[text_mat,embedding.vectors[embedding.stoi[t]].numpy()]
                except:
                    pass
        
        # padding
        if text_mat.shape[1] < self._max_length:
            text_mat = np.c_[np.zeros((300,(self._max_length - text_mat.shape[1]))),text_mat]
        
        return {
                'label': tensor((int(line[9])-1)).long(),
                'text' : tensor(text_mat.T).float()
                }

    def __len__(self):
        return self._total_data
    
def make_dataloaders(path, data_dir, text_seg, batch_size, num_workers):
    train_dataset = MyDataset(f'{path}/{data_dir}/train.txt', text_seg)
    test_dataset = MyDataset(f'{path}/{data_dir}/test.txt', text_seg)
        
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