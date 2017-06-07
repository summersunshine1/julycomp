import os
def listfiles(datadir):
    list_dirs = os.walk(datadir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list