from torch.utils.data import DataLoader,Dataset
import torch

class Dataset(Dataset):
    def __init__(self,data,label,tokenizer):
        self.tokenizer=tokenizer
        self.data=data
        self.label=label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        try:
            x=self.data[idx]
            y=self.label[idx]
        except:
            print(self.data[idx])
        
        encode =self.tokenizer(x, padding="max_length", truncation=True, return_tensors="pt", max_length=256)
        return {
            'input_ids': encode['input_ids'].flatten(),
            'attention_mask':encode['attention_mask'].flatten(),
            'label': torch.tensor(y,dtype=torch.long)
        }    