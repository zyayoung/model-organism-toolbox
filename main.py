import tkinter as tk
import tkinter.filedialog
import os
from nematoda import NematodeTracker, NematodaMovementDetector


def live_nematode_counter():
    file_list = list(tk.filedialog.askopenfilenames())

    for file in file_list:
        if os.path.isdir(file):
            continue
        print('Now processing:', file)
        root_dir, file = os.path.split(file)
        if not os.path.exists(os.path.join(root_dir, 'output')):
            os.mkdir(os.path.join(root_dir, 'output'))
        nematoda = NematodaMovementDetector(
            os.path.join(root_dir, file),
            # Todo: config these parameters in GUI
            resize_ratio=0.5,
            display_scale=1,
        )
        nematoda.config()

        frame_count = nematoda.process(
            online=True,
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


def click_and_track():
    tip = tk.Tk()
    tip.title('Tip')
    t1 = tk.Label(tip, text='Step1: Adjust threshold to make sure nematodes are surround by red lines\n'
                            '       Press Enter to continue...\n'
                            'Step2: Select min & max nematode area\n'
                            'Step3: Left click to select the nematode you want to track\n'
                            '       Right click to deselect\n'
                            '       Press Enter to start Tracking...\n\n'
                            'Output files will be stored in input_video_dir/output',

                  justify=tk.LEFT,)
    t1.pack()
    file_list = list(tkinter.filedialog.askopenfilenames())
    for file in file_list:
        if os.path.isdir(file):
            continue
        print('Now processing:', file)
        nematoda = NematodeTracker(file)
        nematoda.init_threshold()
        nematoda.choose_nematode()
        nematoda.track_nematode()
    tip.destroy()


if __name__ == '__main__':
    window = tk.Tk()
    window.title('Model Organism Toolbox')
    L1 = tk.Label(window, text='Model Organism Toolboxn\nPlease choose an application below!')
    L1.pack()
    B1 = tk.Button(window, text='Live nematoda counter', command=live_nematode_counter, width=25)
    B1.pack()
    B2 = tk.Button(window, text='Nematoda click & track', command=click_and_track, width=25)
    B2.pack()
    window.mainloop()
