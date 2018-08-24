import numpy as np
import cv2
import utils
from video_reader import VideoReader


def nothing(_):
    return

# Todo: Add some comments


class NematodaMovementDetector:
    def __init__(
        self,
        filename=r'D:\Projects\model_organism_helper\Nematoda\capture-0001.avi',
        resize_ratio=1.0,
        frame_step=1,
        movement_threshold=4,
        max_nematoda_count=100,
        kernel_size=None,
        display_scale=1.0,
    ):
        self.frame_step = frame_step
        self.resize_ratio = resize_ratio
        self.video_reader = VideoReader(filename, resize_ratio, frame_step)
        self.background_subtractor = None
        self.movement_threshold = movement_threshold
        self.kernel_size = kernel_size
        if self.kernel_size is None:
            self.kernel_size = int(min(self.video_reader.target_shape) / 32)
            self.kernel_size = int(2 * (int((self.kernel_size - 1) / 2))) + 1

        self.max_nematoda_count = max_nematoda_count
        self.initialize_background_subtractor()
        display_scale = min(display_scale, 400/np.min(self.video_reader.target_shape))
        self.display_size_target = (
            int(self.video_reader.target_shape[0] * display_scale),
            int(self.video_reader.target_shape[1] * display_scale),
        )

    def initialize_background_subtractor(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10)
        for i in range(20):
            frame = self.video_reader.read()
            self.background_subtractor.apply(frame)
            # print('.', end='', flush=True)

    def get_contours(self, frame):
        foreground = cv2.absdiff(self.background_subtractor.getBackgroundImage(), frame)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        foreground = cv2.GaussianBlur(foreground, (self.kernel_size, self.kernel_size), 0)
        _, mask = cv2.threshold(foreground, self.movement_threshold, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def config(self):
        print('Initializing...')
        frame = self.video_reader.read()
        cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
        frame_d = cv2.resize(frame, self.display_size_target)
        labeled_frame_d = cv2.resize(frame, self.display_size_target)
        cv2.imshow('video', np.hstack([frame_d, labeled_frame_d]))
        cv2.createTrackbar('threshold', 'video', self.movement_threshold, 63, nothing)
        cv2.createTrackbar('kernelSize', 'video', self.kernel_size, self.kernel_size * 3, nothing)
        cv2.createTrackbar('frameStep', 'video', self.frame_step, int(self.video_reader.frame_count/20-1), nothing)
        cv2.setTrackbarMin('frameStep', 'video', 1)
        reset_frame_step_countdown = -1
        while True:
            contours = self.get_contours(frame)
            labeled_frame = frame.copy()
            cv2.drawContours(labeled_frame, contours, -1, utils.COLOR['red'], 2, cv2.LINE_AA)
            frame_d = cv2.resize(frame, self.display_size_target)
            labeled_frame_d = cv2.resize(labeled_frame, self.display_size_target)
            cv2.imshow('video', np.hstack([frame_d, labeled_frame_d]))

            self.movement_threshold = cv2.getTrackbarPos('threshold', 'video')
            self.kernel_size = cv2.getTrackbarPos('kernelSize', 'video')
            self.kernel_size = int(2 * (int((self.kernel_size - 1) / 2))) + 1
            if self.frame_step != cv2.getTrackbarPos('frameStep', 'video'):
                self.frame_step = cv2.getTrackbarPos('frameStep', 'video')
                reset_frame_step_countdown = 20

            if reset_frame_step_countdown > 0:
                reset_frame_step_countdown -= 1
            if reset_frame_step_countdown == 0:
                self.video_reader.reset(frame_step=self.frame_step)
                self.initialize_background_subtractor()
                frame = self.video_reader.read()
                reset_frame_step_countdown = -1

            k = cv2.waitKey(30) & 0xff

            if k in [13, 32]:
                break
            elif k == 27:
                cv2.destroyAllWindows()
                exit()
        cv2.destroyAllWindows()
        self.video_reader.reset()
        print('Done')

    def process(self, online=False, output_filename=None):
        # initialize VideoWriter
        wri = None
        if output_filename is not None:
            wri = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                self.video_reader.fps,
                self.video_reader.target_shape,
            )

        _time = cv2.getTickCount()
        frame_count = np.zeros((self.max_nematoda_count,))

        frame = self.video_reader.read()
        frame_idx = 0
        while frame is not None:
            self.background_subtractor.apply(frame)
            contours = self.get_contours(frame)

            if contours:
                if len(contours) < self.max_nematoda_count:
                    frame_count[len(contours)] += 1

            if online or wri is not None:
                labeled_frame = frame.copy()
                cv2.putText(
                    labeled_frame,
                    '%d' % (len(contours),),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    utils.COLOR['yellow'],
                    6
                )
                if contours:
                    cv2.drawContours(labeled_frame, contours, -1, utils.COLOR['red'], 2, cv2.LINE_AA)
                if wri is not None:
                    wri.write(labeled_frame)
                if online:
                    frame = cv2.resize(frame, self.display_size_target)
                    labeled_frame = cv2.resize(labeled_frame, self.display_size_target)
                    cv2.imshow('video', np.hstack([frame, labeled_frame]))
                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break

            frame = self.video_reader.read()
            frame_idx += 1

            # progress report
            if frame_idx % 50 == 0:
                print('%.2f' % (frame_idx * 100.0 / self.video_reader.frame_count) + '%', end=' ')
                time = cv2.getTickCount()
                print(50 / (time - _time) * cv2.getTickFrequency(), 'fps')
                _time = time

        if wri is not None:
            wri.release()
        if online:
            cv2.destroyAllWindows()
            print(frame_count)
            print('prediction:', np.argmax(frame_count))
        return frame_count
