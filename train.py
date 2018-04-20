import inp
import itertools
import json
import os

ID,FORM,LEMMA,FEATS,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)
def build_vocabularies(documents):
    char_vocab={"<PADDING>":0,"<OOV>":1,"<BOS>":2,"<EOS>":3,"<BOD>":4,"<EOD>":5,"<BOW>":6,"<EOW>":7}
    for document,meta in documents:
        for comment,sent in document:
            for cols in sent:
                for char in cols[FORM]:
                    char_vocab.setdefault(char,len(char_vocab))
    return char_vocab

def vectorize_doc(document,char_vocab):
    doc_char_vectorized=[]
    for comment,sent in document:
        sent_char_vectorized=[]
        for cols in sent:
            sent_char_vectorized.append(list(char_vocab.get(char,1) for char in cols[FORM]))
        doc_char_vectorized.append(sent_char_vectorized)
    return doc_char_vectorized



if __name__=="__main__":
    if not os.path.exists("char_vocab.json"):
        docs=inp.get_documents("/home/ginter/text-generation/all_stt.conllu.gz")
        char_vocab=build_vocabularies(itertools.islice(docs,10000))
        with open("char_vocab.json","wt") as f:
            json.dump(char_vocab,f)
    else:
        with open("char_vocab.json","rt") as f:
            char_vocab=json.load(f)

    for d,meta in inp.get_documents("/home/ginter/text-generation/all_stt.conllu.gz"):
        print(vectorize_doc(d,char_vocab))
        break
    
        
