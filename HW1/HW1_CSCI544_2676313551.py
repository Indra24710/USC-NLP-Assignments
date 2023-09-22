cimport warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet',quiet=True)
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4',quiet=True)
nltk.download("stopwords",quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



org_df=pd.read_csv("./data.tsv",sep='\t',on_bad_lines="skip",verbose=False)
required_columns=['review_body','star_rating']
df=org_df[required_columns]
new_df=pd.DataFrame({"review_body":[],"star_rating":[]})
for i in range(1,6):
    new_df=pd.concat([new_df,df.loc[df['star_rating']==i].sample(50000)])



charcount_init=0
charcount_final=0



import re
remove_html_tags='<.*?>';
remove_urls='http\S+';
remove_non_alpha='[^A-Za-z ]'
remove_extra_space=' +'
nones=0
processed={"review_body":[],"star_rating":[]}
for i in range(len(new_df)):
    s=str(new_df['review_body'].iloc[i])
    c=new_df['star_rating'].iloc[i]
    charcount_init+=len(s)
    if s=="":
        continue
    s=re.sub(remove_html_tags,"",s)
    s=re.sub(remove_urls,"",s)
    s=re.sub(remove_non_alpha,"",s)
    s=re.sub(remove_extra_space," ",s)
    if s=="":
        continue
    processed["review_body"].append(contractions.fix(s.lower()))
    charcount_final+=len(processed["review_body"][-1])
    processed["star_rating"].append(int(c))


print(charcount_init,",",charcount_final)


pre_processcount=0
stop_words = set(stopwords.words('english'))
for i in range(len(processed['review_body'])):
    s=processed['review_body'][i]
    pre_processcount+=len(s)
    s=s.split(" ")
    s=[word for word in s if word not in stop_words]
    s=" ".join(s)
    processed['review_body'][i]=s



ps = PorterStemmer()

for i in range(len(processed['review_body'])):
    s=processed['review_body'][i]
    s=s.split(" ")
    s=[ps.stem(word) for word in s]
    s=" ".join(s)
    processed["review_body"][i]=s



lemmatizer = WordNetLemmatizer()

post_processcount=0
count={1:0,2:0,3:0,4:0,5:0}
final_processed_text={"input":[],"target":[]}
for i in range(len(processed['review_body'])):
    s=processed['review_body'][i]
    c=processed['star_rating'][i]
    if count[int(c)]>=20000:
        continue
    s=s.split(" ")
    s=[lemmatizer.lemmatize(word) for word in s]
    if len(s)<13:
        continue
    s=" ".join(s)
    post_processcount+=len(s)
    final_processed_text["input"].append(s)
    final_processed_text["target"].append(int(c))
    count[int(c)]+=1


print(pre_processcount,",",post_processcount)

tfidf = TfidfVectorizer()
vector_rep=tfidf.fit_transform(final_processed_text['input'])

Xtrain,Xtest,ytrain,ytest=train_test_split(vector_rep,final_processed_text['target'],shuffle=True,test_size=0.2,random_state=1)



# #Perceptron

perceptron=Perceptron()
perceptron.fit(Xtrain,ytrain)
percept_pred=perceptron.predict(Xtest)

# # SVM

s=LinearSVC()
s.fit(Xtrain,ytrain)
s_pred=s.predict(Xtest)

# # Logistic Regression


log_reg=LogisticRegression(max_iter=400)
log_reg.fit(Xtrain,ytrain)
log_reg_pred=log_reg.predict(Xtest)


# # Naive Bayes


nb=MultinomialNB()
nb.fit(Xtrain,ytrain)
nb_pred=nb.predict(Xtest)


def getreportvalues(rep):
    rep=rep.split("\n")
    rep=[i.split(" ") for i in rep]
    fin_rep=[]
    for i in rep:
        if i!=[]:
            temp=[]
            for j in i:
                if j!='':
                    temp.append(j)
            if temp!=[]:
                fin_rep.append(temp)
    return fin_rep


#classification reports
percept=classification_report(ytest,percept_pred)
svm_rep=classification_report(ytest,s_pred)
log_reg_rep=classification_report(ytest,log_reg_pred)
nb_rep=classification_report(ytest,nb_pred)


##perceptron report

rep=getreportvalues(percept)
for i in range(1,6):
    s=",".join(rep[i][1:4])
    print(s)
print(",".join(rep[-2][2:5]))
##svm report
rep=getreportvalues(svm_rep)
for i in range(1,6):
    s=",".join(rep[i][1:4])
    print(s)
print(",".join(rep[-2][2:5]))

##logistic regression report
rep=getreportvalues(log_reg_rep)
for i in range(1,6):
    s=",".join(rep[i][1:4])
    print(s)
print(",".join(rep[-2][2:5]))


## naive bayes report
rep=getreportvalues(nb_rep)
for i in range(1,6):
    s=",".join(rep[i][1:4])
    print(s)
print(",".join(rep[-2][2:5]))