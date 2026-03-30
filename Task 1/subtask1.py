import numpy as np
import cv2

from lucaskanade import LucasKanadeTracker

class ImageHelper:
    def draw_rectangle(self, img, xy, fill=None, outline=None):
        if len(xy) == 2 and isinstance(xy[0], (list, tuple)):
            pt1 = (int(xy[0][0]), int(xy[0][1]))
            pt2 = (int(xy[1][0]), int(xy[1][1]))
        else:
            pt1 = (int(xy[0]), int(xy[1]))
            pt2 = (int(xy[2]), int(xy[3]))
        if fill is not None: cv2.rectangle(img, pt1, pt2, color=fill, thickness=-1)
        if outline is not None: cv2.rectangle(img, pt1, pt2, color=outline, thickness=1)
        return img


class VideoMotionTracker:
    def __init__(self, video_path, width=600, height=600, box_size=15):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.box_size = box_size

        self.lk_tracker = LucasKanadeTracker(num_pyramid_levels=2, num_iterations=1)
        self.img_helper = ImageHelper()
        self.kernel = np.ones((5, 5), np.uint8)

        # Tracking state
        self.x0_arr = []
        self.y0_arr = []
        self.velocity_arr = []
        self.stagnation_arr = []
        self.tracks_arr = []

        # Configuration
        self.refresh_interval = 4
        self.max_track_length = 25      # Longer trail window
        self.stagnation_limit = 40
        self.max_corners = 200
        self.min_feature_distance = 10
        self.min_new_point_gap = 12
        self.quality_level = 0.005
        self.motion_threshold = 8
        self.velocity_threshold = 0.05

    def add_tracking_point(self, x, y):
        self.x0_arr.append(x)
        self.y0_arr.append(y)
        self.velocity_arr.append(None)
        self.stagnation_arr.append(0)
        self.tracks_arr.append([])

    def remove_tracking_point(self, index):
        self.x0_arr.pop(index)
        self.y0_arr.pop(index)
        self.velocity_arr.pop(index)
        self.stagnation_arr.pop(index)
        self.tracks_arr.pop(index)

    def _seed_corners(self, gray_frame, motion_mask):
        return cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_feature_distance,
            mask=motion_mask,
        )

    def _motion_mask(self, prev_gray, curr_gray):
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return cv2.dilate(mask, self.kernel, iterations=3)

    def seek_initial_movement(self, cap):
        ret, frame_t0_color = cap.read()
        if not ret:
            return None
        frame_t0_gray = cv2.cvtColor(
            cv2.resize(frame_t0_color, (self.width, self.height)), cv2.COLOR_BGR2GRAY
        )
        print("Scanning video for initial movement...")

        while True:
            ret, frame_t1_color = cap.read()
            if not ret:
                print("No movement found.")
                return None

            frame_t1_gray = cv2.cvtColor(
                cv2.resize(frame_t1_color, (self.width, self.height)), cv2.COLOR_BGR2GRAY
            )
            mask = self._motion_mask(frame_t0_gray, frame_t1_gray)
            corners = self._seed_corners(frame_t1_gray, mask)

            if corners is not None and len(corners) > 0:
                print(f"Movement detected — {len(corners)} features found.")
                for corner in corners:
                    x_c, y_c = corner.ravel()
                    tx = int(x_c - self.box_size / 2)
                    ty = int(y_c - self.box_size / 2)
                    if (0 <= tx <= self.width - self.box_size) and (
                        0 <= ty <= self.height - self.box_size
                    ):
                        self.add_tracking_point(tx, ty)
                return frame_t1_gray

            frame_t0_gray = frame_t1_gray.copy()

    def allocate_new_points(self, prev_gray, curr_gray):
        mask = self._motion_mask(prev_gray, curr_gray)
        corners = self._seed_corners(curr_gray, mask)
        if corners is None:
            return
        for corner in corners:
            x_c, y_c = corner.ravel()
            tx = int(x_c - self.box_size / 2)
            ty = int(y_c - self.box_size / 2)
            if not (
                (0 <= tx <= self.width - self.box_size)
                and (0 <= ty <= self.height - self.box_size)
            ):
                continue
            too_close = any(
                np.hypot(tx - ex, ty - ey) < self.min_new_point_gap
                for ex, ey in zip(self.x0_arr, self.y0_arr)
            )
            if not too_close:
                self.add_tracking_point(tx, ty)

    def lkmath(self, reduced_map_t1, reduced_map_t2, display_frame):
        for i in range(len(self.x0_arr) - 1, -1, -1):
            curr_x = self.x0_arr[i]
            curr_y = self.y0_arr[i]

            out_of_bounds = (
                curr_x <= 0
                or curr_y <= 0
                or curr_x >= self.width - self.box_size - 1
                or curr_y >= self.height - self.box_size - 1
            )
            if out_of_bounds or self.stagnation_arr[i] > self.stagnation_limit:
                self.remove_tracking_point(i)
                continue

            self.velocity_arr[i] = self.lk_tracker.pyramidal_lucas_kanade(
                reduced_map_t1, reduced_map_t2, curr_y, curr_x, self.box_size
            )

            if self.velocity_arr[i] is not None:
                vx = self.velocity_arr[i][0][0]
                vy = self.velocity_arr[i][1][0]
                if abs(vx) > self.velocity_threshold or abs(vy) > self.velocity_threshold:
                    self.x0_arr[i] = int(curr_x + round(vx))
                    self.y0_arr[i] = int(curr_y + round(vy))
                    self.stagnation_arr[i] = 0
                else:
                    self.stagnation_arr[i] += 1

            cx = int(self.x0_arr[i] + self.box_size / 2)
            cy = int(self.y0_arr[i] + self.box_size / 2)
            self.tracks_arr[i].append((cx, cy))
            if len(self.tracks_arr[i]) > self.max_track_length:
                self.tracks_arr[i].pop(0)

            track = self.tracks_arr[i]
            if len(track) < 2:
                continue

            tail = track[0]
            tip  = track[-1]

            disp = np.hypot(tip[0] - tail[0], tip[1] - tail[1])
            if disp < 2.5:
                continue

            brightness = int(min(255, 100 + disp * 12))
            color = (0, 0, brightness)

            pts = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=False, color=color, thickness=2)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_t1_gray = self.seek_initial_movement(cap)
        if frame_t1_gray is None:
            return

        frame_counter = 0

        while True:
            ret, frame_t2_color = cap.read()
            if not ret:
                break

            frame_t2_color = cv2.resize(frame_t2_color, (self.width, self.height))
            frame_t2_gray = cv2.cvtColor(frame_t2_color, cv2.COLOR_BGR2GRAY)

            reduced_map_t1 = frame_t1_gray.astype(float)
            reduced_map_t2 = frame_t2_gray.astype(float)

            frame_counter += 1

            if frame_counter % self.refresh_interval == 0:
                self.allocate_new_points(frame_t1_gray, frame_t2_gray)

            self.lkmath(reduced_map_t1, reduced_map_t2, frame_t2_color)

            cv2.imshow("Lucas-Kanade Optical Flow Tracker", frame_t2_color)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

            frame_t1_gray = frame_t2_gray.copy()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VideoMotionTracker("/home/CJ.mp4")
    app.run()