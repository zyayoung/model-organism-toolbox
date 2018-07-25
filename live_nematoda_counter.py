import os
import glob
from nematoda import Nematoda
import sys
import tkinter.filedialog
import tkinter

if __name__ == '__main__':
    file_list = ['videos/capture-0004.mp4']

    for file in file_list:
        if os.path.isdir(file):
            continue
        print('Now processing:', file)
        root_dir, file = os.path.split(file)
        if not os.path.exists(os.path.join(root_dir,'output')):
            os.mkdir(os.path.join(root_dir,'output'))
        nematoda = Nematoda(
            os.path.join(root_dir, file),
            # Todo: config these parameters in GUI
            resize_ratio=0.5,
            display_scale=1,
        )
        # nematoda.config()

        frame_count = nematoda.process(
            online=False,
            output_filename=os.path.join(os.path.join(root_dir, 'output'), file)
        )

        # write csv file
        if not os.path.exists(os.path.join(os.path.join(root_dir, 'output/results.csv'))):
            f = open(os.path.join(os.path.join(root_dir, 'output/results.csv')), 'w')
            f.write('filename,result,resizeRatio,threshold,kernelSize,frameStep')
            for n_nematoda in range(nematoda.max_nematoda_count):
                f.write(',%d' % n_nematoda)
            f.write('\n')
            f.close()
        f = open(os.path.join(os.path.join(root_dir, 'output/results.csv')), 'a')
        f.write('%s,%s,%f,%d,%d,%d' % (
            file,
            frame_count.argmax(),
            nematoda.resize_ratio,
            nematoda.movement_threshold,
            nematoda.kernel_size,
            nematoda.frame_step,
        ))
        for n_nematoda in range(nematoda.max_nematoda_count):
            f.write(',%d' % frame_count[n_nematoda])
        f.write('\n')
        f.close()
# os.system('PAUSE')
