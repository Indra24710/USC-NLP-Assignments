class HMM:
    def __init__(self):
        self.words={}
        self.postags={}
        self.prevpos={}
        self.transprob={}
        self.openclasstags=[]
    
    def split(self,word):
        j=len(word)-1
        while word[j]!='/':
            j-=1
        return word[:j],word[j+1:]
        
    def create_corpora(self,trainset):
        
        for sentence in trainset:
            sentence=sentence.decode("utf-8")
            sentence=sentence.rstrip()
            wordlist=sentence.split(" ")
            prev='start'
            if 'start' not in self.postags:
                self.postags['start']=[0,[]]
            self.postags['start'][0]+=1
            for word in wordlist:
                w,tag=self.split(word)
                if w not in self.words:
                    self.words[w]={}
                    self.words[w]['tcount']=0
                if tag not in self.words[w]:
                    self.words[w][tag]=0
                if tag not in self.postags:
                    self.postags[tag]=[0,[]]
                    
                self.words[w][tag]+=1
                self.words[w]['tcount']+=1
                self.postags[tag][0]+=1
                self.postags[tag][1].append(w)
                if prev:
                    if prev not in self.prevpos:
                        self.prevpos[prev]={}
                        self.transprob[prev]={}
                    if tag not in self.prevpos[prev]:
                        self.prevpos[prev][tag]=0
                        self.transprob[prev][tag]=0
                    self.transprob[prev][tag]=(self.prevpos[prev][tag]+1)/(self.postags[prev][0]+len(self.postags.keys()))
                    
                    self.prevpos[prev][tag]+=1
                    
                prev=tag
                
    
    def predict_tags(self,sentence):
        sentence=sentence.decode("utf-8")
        sentence=sentence.rstrip()
        sentence=sentence.split(" ")
        output_tags=[]
        outputs=[]
        prev='start'
        for word in sentence:
            
            if word not in self.words:
                m_transprob=float("-inf")
                m_transprob_ind=[]
                tags=self.openclasstags
                for tag in tags:
                    if tag not in self.transprob[prev]:
                        prob=0
                    else:
                        prob=self.transprob[prev][tag]
                    if prob>m_transprob:
                        m_transprob=prob
                        m_transprob_ind=[tag,prob]
                output_tags.append(m_transprob_ind[0])
                outputs.append(word+"/"+m_transprob_ind[0])
                prev=m_transprob_ind[0]
                
            else:
                
                m_transprob=float("-inf")
                m_transprob_ind=[]
                tags=self.prevpos[prev].keys()
                trans_prob=[]
                for tag in tags:
                    prob=self.transprob[prev][tag]
                    if prob>m_transprob:
                        m_transprob=prob
                        m_transprob_ind=[tag,prob]
                    trans_prob.append([tag,prob])
                m_em_prob=float("-inf")
                m_em_prob_ind=[]
                for i in trans_prob:
                    tag,prob=i
                    if tag not in self.words[word].keys():
                        em_prob=0
                        if em_prob>m_em_prob:
                            m_em_prob=em_prob
                            m_em_prob_ind=[tag,em_prob]
                    else:
                        em_prob=self.words[word][tag]/self.postags[tag][0]
                        em_prob*=prob
                        if em_prob>m_em_prob:
                            m_em_prob=em_prob
                            m_em_prob_ind=[tag,em_prob]
                output_tags.append(m_em_prob_ind[0])
                outputs.append(word+"/"+m_em_prob_ind[0])
                prev=m_em_prob_ind[0]
                
        return " ".join(outputs)
                            