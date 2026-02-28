import os

ret = os.system("python src/drift.py")

if ret != 0:
    print("drift detected -> retrain")
    os.system("python src/train.py")
else:
    print("no drift")