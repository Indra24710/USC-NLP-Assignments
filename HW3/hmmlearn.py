import sys
import json
import HMM
                            
args=sys.argv
path=args[1]
hmm_learner=HMM()
f=open(path,'rb')
hmm_learner.create_corpora(f.readlines())
open_class_tags=[[i,len(set(hmm_learner.postags[i][1]))] for i in hmm_learner.postags.keys()]
open_class_tags=sorted(open_class_tags,key=lambda x:x[1])[::-1]
open_class_tags=[i[0] for i in open_class_tags]
hmm_learner.openclasstags=open_class_tags[:5]
model={'words':hmm_learner.words,'tags':hmm_learner.postags,'transitions':hmm_learner.prevpos,"trans_prob":hmm_learner.transprob,"open_class":hmm_learner.openclasstags}
with open('hmmmodel.txt','w',encoding='utf-8') as file:
    json.dump(model,file,ensure_ascii=False)