from glob import glob
import shutil
for path in glob("./data/**/*super-vns*.json", recursive=True):
    print(path)
    shutil.copy(path, "./result_share/")


