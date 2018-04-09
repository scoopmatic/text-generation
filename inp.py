import sys

def get_documents(fname):
    with open(fname) as f:
        current_doc=[]
        current_metadata=None
        for line in f:
            line=line.strip()
            if not line:
                if current_doc:
                    yield current_doc, current_metadata
                    current_doc=[]
                    current_metadata=None
                continue
        else:
           if current_doc:
               yield current_doc, current_metadata


