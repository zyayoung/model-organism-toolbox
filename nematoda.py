import numpy as np
import pandas as pd
import cv2
import glob
import os
import utils
from video_reader import VideoReader


def nothing(_):
    return


class NematodaMovementDetector:
    # Todo: Add some comments
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


class NematodaOptFlow:
    def __init__(
        self,
        filename=r'videos\Nematoda\capture-0056.avi',
        resize_ratio=0.5,
        frame_step=1,
    ):
        self.frame_step = frame_step
        self.resize_ratio = resize_ratio
        self.video_reader = VideoReader(filename, resize_ratio, frame_step)

        self.p0 = []

    @staticmethod
    def on_mouse(event, x, y, flag, param):
        if event == 4:
            param.append((x, y))

    def choose_window(self):
        cv2.namedWindow('init')
        cv2.setMouseCallback('init', self.on_mouse, self.p0)
        frame = self.video_reader.read()
        while True:
            cv2.imshow('init', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        self.p0 = np.array(self.p0, dtype=np.float32).reshape(-1, 1, 2)
        cv2.destroyAllWindows()

    def track(self, output_filename):
        wri = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
            self.video_reader.fps,
            self.video_reader.target_shape,
        )
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0, 255, (100, 3))
        old_frame = self.video_reader.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)

        while True:
            frame = self.video_reader.read()
            if frame is None:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **lk_params)
            # 选取好的跟踪点
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            # 画出轨迹
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            wri.write(img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # 更新上一帧的图像和追踪点
            old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        self.video_reader.release()
        wri.release()


class NematodeTracker:
    def __init__(self, filename='videos/capture-0001.mp4'):
        self.filename = filename
        self.video_reader = VideoReader(filename, resize_ratio=0.5, frame_step=1)
        self.first_frame = self.video_reader.read()
        self.choice = []
        self.chosen_nematode_tot_distance = []
        self.colors = np.random.uniform(0, 255, (100, 3))
        self.threshold = 70
        self.min_area = 10
        self.max_area = 20

        self.data = []

    @staticmethod
    def on_mouse(event, x, y, flag, param):
        if event == 4:
            param.append((x, y))

    @staticmethod
    def l2distance(param):
        pos1, pos2 = param
        return (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2

    @staticmethod
    def nothing(param):
        return

    def init_threshold(self):
        cv2.namedWindow('init threshold')
        cv2.createTrackbar('threshold', 'init threshold', self.threshold, 255, self.nothing)
        frame = self.first_frame
        while True:
            self.threshold = cv2.getTrackbarPos('threshold', 'init threshold')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, gray_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
            display_frame = frame.copy()

            _, contours, _ = cv2.findContours(gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(display_frame, contours, -1, (0, 0, 255))

            display_frame = cv2.putText(display_frame, 'Press Enter To Continue...', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('init threshold', display_frame)
            k = cv2.waitKey(30) & 0xff
            if k in [27, 13, 32]:
                cv2.destroyAllWindows()
                break

    def choose_nematode(self):
        cv2.namedWindow('choose nematode')
        cv2.setMouseCallback('choose nematode', self.on_mouse, self.choice)
        cv2.createTrackbar('minArea', 'choose nematode', self.min_area, 100, self.nothing)
        cv2.createTrackbar('maxArea', 'choose nematode', self.max_area, 100, self.nothing)
        frame = self.first_frame
        while True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, gray_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
            display_frame = frame.copy()
            centers = []
            _, contours, _ = cv2.findContours(gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                self.min_area = cv2.getTrackbarPos('minArea', 'choose nematode')**2
                self.max_area = cv2.getTrackbarPos('maxArea', 'choose nematode')**2
                for idx, contour in enumerate(contours):
                    if self.min_area < cv2.contourArea(contour) < self.max_area:
                        cv2.drawContours(display_frame, contours, idx, (0, 0, 255))
                        M = cv2.moments(contour)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)
                        centers.append((cx, cy))

            for idx in range(len(self.choice)):
                center_idx = np.argmin(list(map(self.l2distance, [(self.choice[idx], center) for center in centers])))
                # print(map(self.l2distance, [(pos, center) for center in centers]))
                self.choice[idx] = centers[center_idx]
                cv2.circle(display_frame, self.choice[idx], 5, tuple(self.colors[idx]), -1)
                display_frame = cv2.putText(display_frame, str(idx), self.choice[idx],
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[idx], 2)

            display_frame = cv2.putText(display_frame, 'Press Enter To Start Tracking...', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('choose nematode', display_frame)
            k = cv2.waitKey(30) & 0xff
            if k in [27, 13, 32]:
                self.chosen_nematode_tot_distance = np.zeros(len(self.choice))
                break

    def track_nematode(self):
        output_dir, name = os.path.split(self.filename)
        output_dir = os.path.join(output_dir, 'output')
        wri = cv2.VideoWriter(
            os.path.join(output_dir, name),
            cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
            self.video_reader.fps,
            self.video_reader.target_shape,
        )
        path_frame = np.zeros_like(self.first_frame)
        for i in range(1, self.video_reader.frame_count):
            frame = self.video_reader.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, gray_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
            display_frame = frame.copy()
            centers = []
            _, contours, _ = cv2.findContours(gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for idx, contour in enumerate(contours):
                    if self.min_area < cv2.contourArea(contour) < self.max_area:
                        cv2.drawContours(display_frame, contours, idx, (0, 0, 255))
                        M = cv2.moments(contour)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)
                        centers.append((cx, cy))

            data_point = []
            for idx in range(len(self.choice)):
                center_idx = np.argmin(list(map(self.l2distance, [(self.choice[idx], center) for center in centers])))
                distance = self.l2distance((self.choice[idx], centers[center_idx]))
                self.chosen_nematode_tot_distance[idx] += distance
                if distance < max(self.video_reader.width, self.video_reader.height)/10:
                    self.choice[idx] = centers[center_idx]
                cv2.circle(path_frame, self.choice[idx], 2, self.colors[idx], -1)
                data_point.append((
                    self.choice[idx][0],
                    self.choice[idx][1],
                    distance,
                    self.chosen_nematode_tot_distance[idx],
                ))

                cv2.circle(display_frame, self.choice[idx], 5, self.colors[idx], -1)
                display_frame = cv2.putText(display_frame, str(idx), self.choice[idx],
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[idx], 2)

            self.data.append(data_point)

            display_frame = cv2.bitwise_or(path_frame, display_frame, display_frame)
            cv2.imshow('choose nematode', display_frame)
            wri.write(display_frame)
            k = cv2.waitKey(30) & 0xff
            if k in [27, 13, 32]:
                break

        data = np.array(self.data)
        data = data.reshape((len(data), -1))
        columns = []
        for i in range(len(self.choice)):
            columns.append('n%dx' % i)
            columns.append('n%dy' % i)
            columns.append('n%dspeed' % i)
            columns.append('n%ddistance' % i)
        df = pd.DataFrame(data=data, columns=columns)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        df.to_csv(os.path.join(output_dir, name + '.csv'))
        wri.release()


if __name__ == "__main__":
    nematode_tracker = NematodeTracker(filename='videos/capture-0056.mp4')
    nematode_tracker.init_threshold()
    nematode_tracker.choose_nematode()
    nematode_tracker.track_nematode()

