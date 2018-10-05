import numpy as np
import pandas as pd
import cv2
import os
from video_reader import VideoReader


def nothing(_):
    return


class NematodeTracker:
    def __init__(self, filename='videos/capture-0001.mp4'):
        self.filename = filename
        self.video_reader = VideoReader(filename)
        self.first_frame = self.video_reader.read()
        self.chosen_nematode_pos = []
        self.nematode_count = 0
        self.chosen_nematode_tot_distance = []
        self.colors = np.random.uniform(0, 255, (100, 3))
        self.threshold = 70

        self.max_display_resolution = (1280, 720)
        self.display_resize_ratio = min(
            self.max_display_resolution[0]/self.video_reader.width,
            self.max_display_resolution[1]/self.video_reader.height,
            1,
        )
        self.target_display_shape = (
            int(self.video_reader.width * self.display_resize_ratio),
            int(self.video_reader.height * self.display_resize_ratio),
        )

        self.min_area = 5  # relative area
        self.max_area = 20
        self.ppa = (self.video_reader.height * self.video_reader.width) / 1253376
        self.elements_resize_ratio = np.sqrt((self.video_reader.height * self.video_reader.width) / 1253376)

        self.data = []

    @staticmethod
    def on_mouse(event, x, y, _, param):
        choice, display_resize_ratio = param
        if event == 4:
            choice.append((int(x / display_resize_ratio), int(y / display_resize_ratio)))
        elif event == 5:
            choice_idx = np.argmin(list(map(
                lambda p: (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2,
                [(c, (int(x / display_resize_ratio), int(y / display_resize_ratio))) for c in choice]
            )))
            choice.pop(choice_idx)

    @staticmethod
    def l2distance(param):
        pos1, pos2 = param
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    @staticmethod
    def l2distance2(param):
        pos1, pos2 = param
        return (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2

    @staticmethod
    def get_eccentricity(rect):
        d1 = rect[1][0]
        d2 = rect[1][1]
        dmax = max(d1, d2)
        dmin = min(d1, d2)
        return np.sqrt(dmax**2-dmin**2)/dmax

    def find_nematode(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, gray_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
        display_frame = frame.copy()
        centers = []
        eccentricities = []
        _, contours, _ = cv2.findContours(gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for idx, contour in enumerate(contours):
                if self.min_area * self.ppa < cv2.contourArea(contour) < self.max_area * self.ppa:
                    cv2.drawContours(display_frame, contours, idx, (0, 0, 255), int(max(1, self.elements_resize_ratio)))
                    m = cv2.moments(contour)
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    cv2.circle(display_frame, (cx, cy), 2, (0, 255, 0), -1)
                    centers.append((cx, cy))
                    ellipse = cv2.fitEllipseDirect(contour)
                    display_frame = cv2.ellipse(display_frame, ellipse, (0, 255, 0), 2)
                    eccentricities.append(self.get_eccentricity(ellipse))

        return display_frame, centers, eccentricities

    def init_threshold(self):
        cv2.namedWindow('tracker')

        # Create Track bar
        frame = self.first_frame
        cv2.imshow('tracker', cv2.resize(frame, self.target_display_shape))
        cv2.createTrackbar('threshold', 'tracker', self.threshold, 255, nothing)

        while True:
            self.threshold = cv2.getTrackbarPos('threshold', 'tracker')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, gray_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
            display_frame = frame.copy()

            _, contours, _ = cv2.findContours(gray_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(display_frame, contours, -1, (0, 0, 255), int(max(1, self.elements_resize_ratio)))

            display_frame = cv2.putText(display_frame, 'Press Enter To Continue...', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('tracker',
                       cv2.resize(display_frame, self.target_display_shape, interpolation=cv2.INTER_AREA))
            k = cv2.waitKey(30) & 0xff
            if k in [27, 13, 32]:
                cv2.destroyAllWindows()
                break

    def choose_nematode(self):
        cv2.namedWindow('tracker')
        cv2.setMouseCallback('tracker', self.on_mouse, (self.chosen_nematode_pos, self.display_resize_ratio))
        cv2.createTrackbar('minArea', 'tracker', self.min_area, 100, nothing)
        cv2.createTrackbar('maxArea', 'tracker', self.max_area, 100, nothing)
        frame = self.first_frame
        while True:
            self.min_area = max(5, cv2.getTrackbarPos('minArea', 'tracker') ** 2)
            self.max_area = max(5, cv2.getTrackbarPos('maxArea', 'tracker') ** 2)
            display_frame, centers, eccentricities = self.find_nematode(frame)
            if centers:
                for chosen_nematode_pos_idx in range(len(self.chosen_nematode_pos)):
                    center_idx = int(np.argmin(list(map(
                        self.l2distance2,
                        [(self.chosen_nematode_pos[chosen_nematode_pos_idx], center) for center in centers],
                    ))))
                    self.chosen_nematode_pos[chosen_nematode_pos_idx] = centers[center_idx]
                    cv2.circle(
                        display_frame,
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        int(5 * self.elements_resize_ratio),
                        self.colors[chosen_nematode_pos_idx],
                        -1
                    )
                    cv2.putText(
                        display_frame,
                        str(chosen_nematode_pos_idx),
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.elements_resize_ratio,
                        self.colors[chosen_nematode_pos_idx],
                        int(2 * self.elements_resize_ratio),
                    )

            cv2.putText(display_frame, 'Press Enter To Start Tracking...', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('tracker', cv2.resize(display_frame, self.target_display_shape, interpolation=cv2.INTER_AREA))
            k = cv2.waitKey(30) & 0xff
            if k in [27, 13, 32]:
                self.chosen_nematode_tot_distance = np.zeros(len(self.chosen_nematode_pos))
                cv2.destroyWindow('tracker')
                break

    def track_nematode(self):
        output_dir, name = os.path.split(self.filename)
        output_dir = os.path.join(output_dir, 'output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        wri = cv2.VideoWriter(
            os.path.join(output_dir, name),
            cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
            self.video_reader.fps,
            self.video_reader.target_shape,
        )
        path_layer = np.zeros_like(self.first_frame)
        for i in range(1, self.video_reader.frame_count):
            text_layer = np.zeros_like(self.first_frame)
            display_frame, cur_centers, eccentricities = self.find_nematode(self.video_reader.read())

            data_point = []
            for chosen_nematode_pos_idx in range(len(self.chosen_nematode_pos)):
                if not cur_centers:
                    distance = np.infty
                    cur_center_idx = -1
                else:
                    cur_center_idx = int(np.argmin(list(map(
                        self.l2distance2,
                        [(self.chosen_nematode_pos[chosen_nematode_pos_idx], cur_center) for cur_center in cur_centers],
                    ))))
                    distance = self.l2distance((
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        cur_centers[cur_center_idx],
                    ))

                # display and record data according to whether the selected nematode is tracked correctly
                if distance < max(self.video_reader.width, self.video_reader.height)/10:
                    self.chosen_nematode_pos[chosen_nematode_pos_idx] = cur_centers[cur_center_idx]
                    cv2.circle(
                        path_layer,
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        2,
                        self.colors[chosen_nematode_pos_idx],
                        -1
                    )
                    cv2.circle(
                        display_frame,
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        int(5 * self.elements_resize_ratio),
                        self.colors[chosen_nematode_pos_idx],
                        -1
                    )
                    cv2.putText(
                        text_layer,
                        '%d %d %.2f' % (
                            chosen_nematode_pos_idx,
                            self.chosen_nematode_tot_distance[chosen_nematode_pos_idx],
                            eccentricities[cur_center_idx]
                        ),
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.elements_resize_ratio,
                        self.colors[chosen_nematode_pos_idx],
                        int(2 * self.elements_resize_ratio),
                    )
                else:
                    distance = 0
                    cv2.putText(
                        path_layer,
                        '?',
                        self.chosen_nematode_pos[chosen_nematode_pos_idx],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.elements_resize_ratio,
                        self.colors[chosen_nematode_pos_idx],
                        int(2 * self.elements_resize_ratio),
                    )

                self.chosen_nematode_tot_distance[chosen_nematode_pos_idx] += distance
                data_point.append((
                    self.chosen_nematode_pos[chosen_nematode_pos_idx][0],
                    self.chosen_nematode_pos[chosen_nematode_pos_idx][1],
                    distance,
                    self.chosen_nematode_tot_distance[chosen_nematode_pos_idx],
                    eccentricities[cur_center_idx],
                ))
            self.data.append(data_point)

            # combine information layer with the original video
            display_frame = cv2.bitwise_xor(display_frame, path_layer, display_frame)
            display_frame = cv2.bitwise_xor(display_frame, text_layer, display_frame)
            cv2.putText(display_frame, 'Total nematode cnt: %d' % len(self.chosen_nematode_pos), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('tracker', cv2.resize(display_frame, self.target_display_shape))
            wri.write(display_frame)
            k = cv2.waitKey(1) & 0xff
            if k in [27, 13, 32]:
                break

        # save recorded data
        data = np.array(self.data)
        data = data.reshape((len(data), -1))
        columns = []
        for i in range(len(self.chosen_nematode_pos)):
            columns.append('n%dx' % i)
            columns.append('n%dy' % i)
            columns.append('n%dspeed' % i)
            columns.append('n%ddistance' % i)
            columns.append('n%deccentricity' % i)
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(os.path.join(output_dir, name + '.csv'))
        wri.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    nematode_tracker = NematodeTracker(filename='videos/capture-0056.mp4')
    nematode_tracker.init_threshold()
    nematode_tracker.choose_nematode()
    nematode_tracker.track_nematode()
