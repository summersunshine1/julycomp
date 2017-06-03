import os
import sys

def isFinalDir(rootDir): 
    list_dirs = os.walk(rootDir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for d in dirs:
            if 'datasets' in d:
                return True
    return False

def getparentdir():
    pwd = sys.path[0]
    abs_path = os.path.abspath(pwd)
    temppath = abs_path
    i=0
    while(i<10):
        index = temppath.rfind('\\')
        if index==-1:
            print("cannot find path")
            break;
        temppath = temppath[:index]
        if(isFinalDir(temppath)):
            return temppath
        i+=1;
    print("not find")
    return ""
    
    
if __name__ == "__main__":
    print(getparentdir())
