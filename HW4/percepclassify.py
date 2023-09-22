import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')
import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
import sys
import json
lemmatizer = WordNetLemmatizer()

def preprocess_data(new_df):
    remove_html_tags='<.*?>';
    remove_urls='http\S+';
    remove_non_alpha='[^A-Za-z ]'
    remove_extra_space=' +'
    processed=[]
    new_df=list(new_df)
    for i in range(len(new_df)):
        s=str(new_df[i])
        if s=="":
            continue
        s=re.sub(remove_html_tags,"",s)
        s=re.sub(remove_urls,"",s)
        s=re.sub(remove_non_alpha,"",s)
        s=re.sub(remove_extra_space," ",s)
        if s=="":
            continue
        processed.append(s)
        
        
    stop_words = set(stopwords.words('english'))
    final_processed_text=[]
    
    for i in range(len(processed)):
        s=processed[i]
        s=s.split(" ")
        s=[lemmatizer.lemmatize(word) for word in s if word not in stop_words]
        final_processed_text.append(s)
        
    return final_processed_text


class DataLoader:
    
    def __init__(self):
        
        self.wordmap={}
        self.wordcount=0
        self.occurences={}
        self.total_sentences=0
    
    def create(self,data):
        
        for sentence in data:
            
            for i in sentence+["end"]:
                i=i.lower()
                
                if i not in self.wordmap:
                    self.wordmap[i]=[self.wordcount,0]
                    self.wordcount+=1
                self.wordmap[i][1]+=1
    
    def get_term_frequency(self,sentence,word):
        
        sentence=[i.lower() for i in sentence]
        return sentence.count(word)/len(sentence)
        
                    
                
    def get_inverse_document_frequency(self,word):
        inv_doc_freq=np.log(self.total_sentences/self.wordmap[word][1])
        return inv_doc_freq
                
    def get_embeddings(self,sentence):
        

        temp=[0]*self.wordcount
        for i in sentence:
            i=i.lower()
            if i in self.wordmap:
                temp[self.wordmap[i][0]]=self.get_term_frequency(sentence,i)*self.get_inverse_document_frequency(i)
        
        return temp

def get_word2vec_embeddings(Xtrain,Xtest,fixedsize=0,dataloader=None):
    Xtrain_embeddings=[]
    Xtest_embeddings=[]
    

    for i in Xtrain:
        Xtrain_embeddings.append(dataloader.get_embeddings(i))
        
    for i in Xtest:
            
        Xtest_embeddings.append(dataloader.get_embeddings(i))
    
    
    return Xtrain_embeddings,Xtest_embeddings


class Node:
    
    def __init__(self,w_size=1000,bias=0):
        
        self.w=np.asarray([0]*w_size)
        self.bias=bias
    
    def __init__(self,w_size=1000,bias=0,avg=True):
        self.w=np.asarray([0]*w_size)
        self.bias=bias
        self.cw=np.asarray([0]*w_size)
        self.cbias=bias
        
class Perceptron:
    def __init__(self,model_type='vanilla',emb_size=4500,bias=0,n=2):
        self.emb_size=emb_size
        self.model_type=model_type
        self.perceptrons=[Node(emb_size,bias),Node(emb_size,bias,True)]
        
        
    def forward(self,inp):
        out=[]
        for i in range(len(self.perceptrons)):
            prod=np.dot(inp,self.perceptrons[i].w)
            s=prod+self.perceptrons[i].bias
            out.append(s)
        return out
    
    def train(self,data,label1,label2):

        if self.model_type=='vanilla':

            for inp,lab1,lab2 in zip(data,label1,label2):
                out=self.forward(np.asarray(inp))
                lab=[lab1,lab2]
                for i in range(len(out)):
                    if out[i]*lab[i]<=0:
                        self.perceptrons[i].w=self.perceptrons[i].w+inp*lab[i]
                        self.perceptrons[i].bias=self.perceptrons[i].bias+lab[i]
        else:
            counter=0
            for inp,lab1,lab2 in zip(data,label1,label2):
                out=self.forward(np.asarray(inp))
                lab=[lab1,lab2]
                if out[0]*lab[0]<=0:
                    
                    self.perceptrons[0].w=self.perceptrons[0].w+inp*lab[0]
                    
                    self.perceptrons[0].bias=self.perceptrons[0].bias+lab[0]
                    
#                     self.perceptrons[0].cw=self.perceptrons[0].cw+lab[0]*counter*inp
                    
#                     self.perceptrons[0].cbias=self.perceptrons[0].cbias+lab[0]*counter
                    self.perceptrons[0].cw=self.perceptrons[0].cw+self.perceptrons[0].w
                    
                    self.perceptrons[0].cbias=self.perceptrons[0].cbias+self.perceptrons[0].bias
                    
                    
                
                if out[1]*lab[1]<=0:
                    
                    self.perceptrons[1].w=self.perceptrons[1].w+inp*lab[1]
                    
                    self.perceptrons[1].bias=self.perceptrons[1].bias+lab[1]
                    
#                     self.perceptrons[1].cw=self.perceptrons[1].cw+lab[1]*counter*inp
                    
#                     self.perceptrons[1].cbias=self.perceptrons[1].cbias+lab[1]*counter
                    
                    self.perceptrons[1].cw=self.perceptrons[1].cw+self.perceptrons[1].w
                    
                    self.perceptrons[1].cbias=self.perceptrons[1].cbias+self.perceptrons[1].bias
                    
                counter+=1

            self.perceptrons[0].w=(1/counter)*self.perceptrons[0].cw
            
            self.perceptrons[0].bias=(1/counter)*self.perceptrons[0].cbias
            
            self.perceptrons[1].w=(1/counter)*self.perceptrons[1].cw
            
            self.perceptrons[1].bias=(1/counter)*self.perceptrons[1].cbias

             
                        
    def predict(self,data):
        out=[]
        for i in data:
            out.append(self.forward(i))
        pred1=[]
        pred2=[]
        for i in out:
            temp=[1,1]
            if i[0]<0:
                temp[0]=-1
            if i[1]<0:
                temp[1]=-1
            pred1.append(temp[0])
            pred2.append(temp[1])
        return pred1,pred2


args=sys.argv
modelpath=args[1]
datapath=args[2]
test_f=open(datapath,'r')

test_map={"id":[],"text":[]}
for i in test_f.readlines():
    data=i.split(" ")
    test_map["id"].append(data[0])
    test_map["text"].append(" ".join(data[1:]))

vanillamodel=open(modelpath,'r')
vanillamodel=json.load(vanillamodel)
wordcount=int(vanillamodel['wordcount'])
dataloader=DataLoader()
dataloader.wordcount=wordcount
dataloader.total_sentences=int(vanillamodel['total_sentences'])
dataloader.wordmap=vanillamodel['wordmap']
for i in dataloader.wordmap.keys():
    temp=dataloader.wordmap[i]
    dataloader.wordmap[i]=[int(temp[0]),int(temp[1])]


test_df=pd.DataFrame(test_map)
test_df['text']=preprocess_data(test_df['text'])
Xtest=test_df['text']
_,Xtest_emb=get_word2vec_embeddings(Xtest,Xtest,fixedsize=200,dataloader=dataloader)
Xtest_emb=np.asarray(Xtest_emb)
classmap={"Fake":-1,"Neg":-1,"Pos":1,"True":1}

p1=Perceptron(emb_size=wordcount)
p1_w1=[float(i) for i in vanillamodel['weights_class_1']]
p1_b1=float(vanillamodel['bias_class_1'])
p1_w2=[float(i) for i in vanillamodel['weights_class_2']]
p1_b2=float(vanillamodel['bias_class_2'])
p1.perceptrons[0].w,p1.perceptrons[0].b=p1_w1,p1_b1
p1.perceptrons[1].w,p1.perceptrons[1].b=p1_w2,p1_b2

pred1,pred2=p1.predict(Xtest_emb)
outputs=[]
for i in range(len(Xtest_emb)):
    predclass1='True'
    if pred1[i]==-1:
        predclass1='Fake'
    predclass2='Pos'
    if pred2[i]==-1:
        predclass2='Neg'
    temp=test_df['id'][i]+" "+predclass1+" "+predclass2+"\n"
    outputs.append(temp)
f=open('perceptoutput.txt','w',encoding='utf-8')
for i in outputs:
    f.write(i)
f.close()