import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer,RobertaTokenizer,RobertaModel,DebertaModel
from sentence_transformers import SentenceTransformer,models

class Classifier(nn.Module):
    
    def __init__(self,device,n_labels,emb_model):
        super().__init__()
        
        self.device=device
        self.emb_model=emb_model
        self.fc=nn.Linear(768,n_labels)
        
    def forward(self,ids,xmsk):
        
        result = self.emb_model(ids,xmsk)
        embeddings = result.last_hidden_state[:, 0, :]
        output=self.fc(embeddings)
        
        return output
    
