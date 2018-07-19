import os
import glob
from nematoda import Nematoda
import sys

if __name__ == '__main__':
    if not os.path.exists('results.csv'):
        f = open('results.csv', 'w')
        f.write('filename,result,resizeRatio,threshold,kernelSize,frameStep\n')
        f.close()
    root_dir = r'videos\Nematoda'
    try:
        if not os.path.exists(root_dir+'\output'):
            os.mkdir(root_dir+'\output')
        for file in glob.glob(root_dir+r'\*.*'):
            file = os.path.split(file)[1]
            nematoda = Nematoda(
                os.path.join(root_dir, file),
                resize_ratio=.5,
                display_scale=.5,
            )
            nematoda.config()

        nematoda_cnt = nematoda.process(
            online=True,
            output_filename=os.path.join(os.path.join(root_dir, 'output'), file)
        )

        # write csv file
        if not os.path.exists(os.path.join(os.path.join(root_dir, 'output/results.csv'))):
            f = open(os.path.join(os.path.join(root_dir, 'output/results.csv')), 'w')
            f.write('filename,result,resizeRatio,threshold,kernelSize,frameStep\n')
            f.close()
        f = open(os.path.join(os.path.join(root_dir, 'output/results.csv')), 'a')
        f.write('%s,%s,%f,%d,%d,%d\n' % (
            file,
            nematoda_cnt,
            nematoda.resize_ratio,
            nematoda.movement_threshold,
            nematoda.kernel_size,
            nematoda.frame_step,
        ))
        f.close()
os.system('PAUSE')
