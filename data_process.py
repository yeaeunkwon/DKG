import pandas as pd
from langdetect import detect
import json

def make_abs_to_string(dict_ab):
    dict_ab=json.loads(dict_ab)
    abstract={}
    for word, indices in dict_ab.items():
        for index in indices:
            abstract[index]=word
    abstract=sorted(abstract.items())
    return " ".join(x[1] for x in abstract)


df=pd.read_csv("two_categories_df.csv")
only_df=df[df.apply(lambda x: len(x['fields_of_study'])==1,axis=1)]
inter_df=df[df.apply(lambda x: len(x['fields_of_study'])==2,axis=1)]
only_df['abstract']=only_df.apply(lambda x: make_abs_to_string(x['abstract_inverted_index']),axis=1)
only_df['text']=only_df['title']+only_df['abstract']
inter_df['abstract']=inter_df.apply(lambda x: make_abs_to_string(x['abstract_inverted_index']),axis=1)
inter_df['text']=inter_df['title']+inter_df['abstract']
only_df.dropna(inplace=True)
inter_df.dropna(inplace=True)
langs=[]
for text in only_df['text']:
    if len(text)<50:
        langs.append("na")
        continue
    try:
        lang=detect(text)
        langs.append(lang)
    except:
        langs.append("na")
only_df['lang']=langs

en_only_df=only_df[only_df['lang']=="en"]
en_only_df=en_only_df[['work_id','publication_year','fields_of_study','text','lang']]
en_only_df.to_csv("en_only_df.csv",index=False)

langs=[]
for text in inter_df['text']:
    if len(text)<50:
        langs.append("na")
        continue
    try:
        lang=detect(text)
        langs.append(lang)
    except:
        langs.append("na")
inter_df['lang']=langs

en_inter_df=inter_df[inter_df['lang']=="en"]
en_inter_df=en_inter_df[['work_id','publication_year','fields_of_study','text','lang']]
en_inter_df.to_csv("en_inter_df.csv",index=False)