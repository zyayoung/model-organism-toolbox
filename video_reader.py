import cv2


class VideoReader:
    def __init__(self, filename, resize_ratio=1.0, frame_step=1):
        self.filename = filename
        self.frame_step = frame_step

        self.cap = cv2.VideoCapture(filename)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.target_shape = (int(self.width * resize_ratio), int(self.height * resize_ratio))

    def read(self):
        for _ in range(1, self.frame_step):
            self.cap.read()
        ret, frame = self.cap.read()
        if frame is not None:
            return cv2.resize(frame, self.target_shape)
        else:
            return None

    def reset(self, frame_step=None):
        if frame_step:
            self.frame_step = frame_step
        self.cap = cv2.VideoCapture(self.filename)
