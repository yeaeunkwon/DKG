import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer,AdamW
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
import torch.nn.functional as F 
import random
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer,DebertaModel
from sentence_transformers import SentenceTransformer,models
from specter_model import Classifier
from specter_dataset import Dataset

parser=argparse.ArgumentParser(description='INFOPJ3')
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--lr',type=float,default=1e-5)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--n_label',type=int,default=19)
parser.add_argument('--batch_size',type=int,default=8)
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def test_fn(model,test_dataloader,device):
    
    model.eval()
    pred=[]
    test_label=[]
    probabilities=[]
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            ids=data['input_ids'].to(device)
            xmsk=data['attention_mask'].to(device)
            label=data['label'].to(device)
            output=model(ids,xmsk)
        
            
            pred.extend(torch.argmax(F.softmax(output),axis=1).detach().cpu().numpy())
            test_label.append(label.detach().cpu().numpy())  
            probabilities.append(F.softmax(output))
        

    return pred,probabilities

def val_fn(valid_dataloader,model,device,criterion,valid_label):
    
    model.eval()
    valid_loss=0.0
    pred=[]
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            ids=data['input_ids'].to(device)
            xmsk=data['attention_mask'].to(device)
            label=data['label'].to(device)
            output=model(ids,xmsk)
            loss=criterion(output,label)
            valid_loss+=loss.item()
            
            pred.extend(torch.argmax(F.softmax(output),axis=1).detach().cpu().numpy())
        
        print(f"valud_pred: {pred}")    
        f1=f1_score(pred,valid_label,average="micro")
        acc=accuracy_score(pred,valid_label)
        report=classification_report(pred,valid_label)
        avg_valid_loss=valid_loss/len(valid_dataloader)
    return f1,acc,report,avg_valid_loss

def train_fn(train_dataloader,model,device,optimizer,criterion):
    
    model.train()
    train_loss=0.0
    for i,data in enumerate(train_dataloader):
        ids=data['input_ids'].to(device)
        xmsk=data['attention_mask'].to(device)
        label=data['label'].to(device)
        output=model(ids,xmsk)
        loss=criterion(output,label)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        if i%10==0:
            print(f"{i} : train_loss {train_loss}")
        
    avg_train_loss=train_loss/len(train_dataloader)
    return model,avg_train_loss
          
def experiment_fn(train_dataloader,valid_dataloader,device,valid_label,test_dataloader):
    
    model=Classifier(device,args.n_label,emb_model).to(device)
    optimizer=AdamW(model.parameters(),lr=args.lr,eps=1e-8)
    criterion=nn.CrossEntropyLoss().to(device)
    best_loss=0
    for i,ep in enumerate(range(args.epoch)):
        model,avg_train_loss=train_fn(train_dataloader,model,device,optimizer,criterion)
        
        f1,acc,report,avg_valid_loss=val_fn(valid_dataloader,model,device,criterion,valid_label)
        print(f"EP:{ep} | avg_train_loss : {avg_train_loss} | avg_valid_loss : {avg_valid_loss} | f1 : {f1} | acc : {acc} | report : {report}")
        if i==0 or best_loss>avg_valid_loss:
            best_loss=avg_valid_loss
            save_dict={"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict()}
            m_name=f"specter_{ep}"
            torch.save(save_dict,"/home/labuser/Applied_NLP/KG/"+m_name+".pt")  
            
    #test
    model_path="/home/labuser/Applied_NLP/KG/"+m_name+".pt"
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    predictions,prob=test_fn(model,test_dataloader,device)
    return predictions,prob
        
if __name__=="__main__":
        
    set_seed(args.seed) 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model=AutoModel.from_pretrained('allenai/specter').to(device)
    tokenizer=AutoTokenizer.from_pretrained('allenai/specter')

   

    data=pd.read_csv("final_train_df.csv")
    text=data['text']
    labels=data['fields_of_study']
    le = LabelEncoder()
    labels=le.fit_transform(labels)
    true_classes = {i:c for i,c in enumerate(le.classes_)}
    print(true_classes)
    train_text,valid_text,train_label,valid_label=train_test_split(list(text),labels,test_size=0.2,random_state=args.seed,stratify=labels)
    print(len(train_text),len(valid_text))
    
    train_dataset=Dataset(train_text,train_label,tokenizer)
    valid_dataset=Dataset(valid_text,valid_label,tokenizer)

    train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size)
    valid_dataloader=DataLoader(valid_dataset,shuffle=False,batch_size=args.batch_size)
    test_df=pd.read_csv("final_test_data.csv")
    test_text=test_df['text']
    test_label=test_df['label']
    
    test_dataset=Dataset(test_text,test_label,tokenizer)
    test_dataloader=DataLoader(test_dataset,shuffle=False,batch_size=1)
    predictions,prob=experiment_fn(train_dataloader,valid_dataloader,device,valid_label,test_dataloader)
    test_df['predicted_class']=predictions
    test_df['probabilities']=prob
    
    test_df.to_csv("test_results.csv",index=False)
    
