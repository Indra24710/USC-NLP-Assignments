import sys
import json
import HMM
                            
                            
args=sys.argv
model=None
with open('hmmmodel.txt','r',encoding='utf-8') as file:
    model=json.load(file)
    
hmm_decoder=HMM()
hmm_decoder.words=model['words']
hmm_decoder.postags=model['tags']
hmm_decoder.prevpos=model['transitions']
hmm_decoder.transprob=model['trans_prob']
hmm_decoder.openclasstags=model['open_class']
f=open(args[1],'rb')
output=[]
for sentence in f.readlines():
    output.append(hmm_decoder.predict_tags(sentence)+'\n')

f=open('hmmoutput.txt','w',encoding='utf-8')
for i in output:
    f.write(i)
f.close()
    
    