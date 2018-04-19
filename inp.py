import sys

def is_new_doc(comments):
    for c in comments:
        if c.startswith("# fname"):
            return True
    return False

def get_documents(fname):
    if fname.endswith(".gz"):
        import gzip
        f=gzip.open(fname, "rt")
    else:
        f=open(fname)
    current_doc=[]
    current_metadata=None
    for comm, sent in get_sentences(f):
        if comm and is_new_doc(comm):  # new document starts
            if current_doc:
                yield current_doc, current_metadata
            current_doc, current_metadata = [], None
        current_doc.append((comm,sent))
    else:
        if current_doc:
            yield current_doc, current_metadata

    f.close()


def get_sentences(f):
    """conllu reader"""
    sent=[]
    comment=[]
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comment,sent
            comment=[]
            sent=[]
        elif line.startswith("#"):
            comment.append(line)
        else: #normal line
            sent.append(line.split("\t"))
    else:
        if sent:
            yield comment, sent


if __name__=="__main__":

    # test
    for i, (document, doc_meta) in enumerate(get_documents("/home/ginter/text-generation/all_stt.conllu.gz")):
        print("Document:",i)
        for comm,sent in document:
            if comm:
                print(comm)
            print(sent)
        print()
        break


