import os
import re
def last_file_num(path):
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    filelist=os.listdir(path)
    
    max_num=0
    if len(filelist)%30000==0:
        return 1
    else:
        pattern=f"\\_(\\d+)\\."
        print(filelist)
        for file in filelist:
            numbers = re.findall(pattern, file)
            print(numbers)
            num=int(numbers[0])
            if num>max_num:
                max_num=num
        return max_num+1