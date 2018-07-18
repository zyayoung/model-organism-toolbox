import os
import glob
from nematoda import Nematoda

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

            results = open('results.csv', 'a')
            nematoda_cnt = nematoda.process(
                online=True,
                output_filename=os.path.join(os.path.join(root_dir, 'output'), file)
            )
            results.write('%s,%s,%f,%d,%d,%d\n' % (
                file,
                nematoda_cnt,
                nematoda.resize_ratio,
                nematoda.movement_threshold,
                nematoda.kernel_size,
                nematoda.frame_step,
            ))
            results.close()
    except InterruptedError:
        print('bye')
