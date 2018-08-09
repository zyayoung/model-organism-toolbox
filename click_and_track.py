import os
from nematoda import NematodeTracker
import sys
import tkinter.filedialog
import tkinter

if __name__ == '__main__':
    file_list = []
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        file_list = list(tkinter.filedialog.askopenfilenames())

    for file in file_list:
        if os.path.isdir(file):
            continue
        print('Now processing:', file)
        nematoda = NematodeTracker(file)
        nematoda.init_threshold()
        nematoda.choose_nematode()
        nematoda.track_nematode()
