import numpy as np
import cv2


class DenseOpticalFlowManual:
    """
    Dense optical flow using manual Lucas-Kanade on a full-frame grid.

    Visualization:
      - HSV color wheel overlay  : hue = flow direction, value = magnitude
      - Arrow grid               : arrows at every grid point, scaled by speed
      - Both are blended onto the original frame

    This correctly handles the case where motion is localized (e.g. a fast
    moving car) AND global (camera pan) — because flow is computed everywhere,
    not just in detected motion regions.
    """

    def __init__(self, video_path, width=832, height=464,
                 grid_step=8, window_size=21,
                 arrow_grid=32, hsv_alpha=0.55, arrow_alpha=0.9):
        """
        Args:
            video_path   : path to the video file
            width/height : display resolution
            grid_step    : LK computation grid spacing (pixels)
            window_size  : LK window size
            arrow_grid   : arrow grid spacing (can be coarser than LK grid)
            hsv_alpha    : blend weight of HSV overlay (0=invisible, 1=full)
            arrow_alpha  : blend weight of arrow overlay
        """
        self.video_path  = video_path
        self.width       = width
        self.height      = height
        self.grid_step   = grid_step
        self.window_size = window_size
        self.arrow_grid  = arrow_grid
        self.hsv_alpha   = hsv_alpha
        self.arrow_alpha = arrow_alpha

        # Pre-built grid of (row, col) sample positions
        self._ys = np.arange(0, self.height, self.grid_step)
        self._xs = np.arange(0, self.width,  self.grid_step)
        self._gh = len(self._ys)
        self._gw = len(self._xs)

    # ------------------------------------------------------------------ gradients
    def _compute_gradients(self, f1: np.ndarray, f2: np.ndarray):
        """Sobel spatial gradients on f1, temporal gradient between f1 and f2.
        Both frames are pre-blurred to suppress noise before differentiation,
        mirroring what Farneback does internally with its polynomial expansion."""
        f1f = cv2.GaussianBlur(f1.astype(np.float32), (5, 5), 1.2)
        f2f = cv2.GaussianBlur(f2.astype(np.float32), (5, 5), 1.2)
        Ix = cv2.Sobel(f1f, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(f1f, cv2.CV_32F, 0, 1, ksize=3)
        It = f2f - f1f
        return Ix, Iy, It

    # ------------------------------------------------------------------ LK at one point
    def _lk_point(self, Ix, Iy, It, row: int, col: int) -> tuple:
        """
        Solve the 2x2 Lucas-Kanade normal equations at (row, col).
        Returns (vx, vy).
        """
        h, w = Ix.shape
        half  = self.window_size // 2
        r0, r1 = max(0, row - half), min(h, row + half + 1)
        c0, c1 = max(0, col - half), min(w, col + half + 1)

        ix = Ix[r0:r1, c0:c1].ravel()
        iy = Iy[r0:r1, c0:c1].ravel()
        it = It[r0:r1, c0:c1].ravel()

        # Normal equations: A^T A v = -A^T b
        Axx = float(np.dot(ix, ix))
        Axy = float(np.dot(ix, iy))
        Ayy = float(np.dot(iy, iy))
        bx  = float(np.dot(ix, it))
        by  = float(np.dot(iy, it))

        det = Axx * Ayy - Axy * Axy
        if abs(det) < 1e-6:
            return 0.0, 0.0

        vx = (-Ayy * bx + Axy * by) / det
        vy = ( Axy * bx - Axx * by) / det
        return vx, vy

    # ------------------------------------------------------------------ dense flow
    def compute_dense_flow(self, f1_gray: np.ndarray, f2_gray: np.ndarray) -> np.ndarray:
        """
        Compute optical flow at every grid point, bilinearly upscale to full res.

        Returns:
            flow: float32 array of shape (H, W, 2)  — channel 0=vx, 1=vy
        """
        Ix, Iy, It = self._compute_gradients(f1_gray, f2_gray)

        sparse_vx = np.zeros((self._gh, self._gw), np.float32)
        sparse_vy = np.zeros((self._gh, self._gw), np.float32)

        for gi, y in enumerate(self._ys):
            for gj, x in enumerate(self._xs):
                vx, vy = self._lk_point(Ix, Iy, It, int(y), int(x))
                sparse_vx[gi, gj] = vx
                sparse_vy[gi, gj] = vy

        # Clamp outliers before upscaling
        sparse_vx = np.clip(sparse_vx, -30, 30)
        sparse_vy = np.clip(sparse_vy, -30, 30)

        # Upscale sparse grid → full resolution (correct: regular grid → cv2.resize)
        flow_vx = cv2.resize(sparse_vx, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        flow_vy = cv2.resize(sparse_vy, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Post-smooth the flow field — this is the key step that makes blobs
        # continuous and removes the patchy look. Farneback achieves this via
        # its polynomial expansion; we replicate it with a Gaussian blur.
        flow_vx = cv2.GaussianBlur(flow_vx, (21, 21), 5.0)
        flow_vy = cv2.GaussianBlur(flow_vy, (21, 21), 5.0)

        return np.stack([flow_vx, flow_vy], axis=2)

    # ------------------------------------------------------------------ HSV overlay
    def _flow_to_hsv(self, flow: np.ndarray) -> np.ndarray:
        """
        Convert flow field to an HSV image (BGR output) for overlay.
          hue        = direction (0–180 maps to 0–360°)
          saturation = 255 (fully saturated)
          value      = magnitude, normalised globally per frame
        """
        vx, vy   = flow[..., 0], flow[..., 1]
        mag, ang = cv2.cartToPolar(vx, vy)

        # Normalise magnitude to 0-255, boosted so slow motion is still visible
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Boost faint signals
        mag_norm = np.clip(mag_norm.astype(np.uint16) * 3, 0, 255).astype(np.uint8)

        hsv        = np.zeros((self.height, self.width, 3), np.uint8)
        hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)  # hue
        hsv[..., 1] = 255                                              # saturation
        hsv[..., 2] = mag_norm                                         # value

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ------------------------------------------------------------------ arrow grid
    def _draw_arrows(self, canvas: np.ndarray, flow: np.ndarray):
        """
        Draw a grid of flow arrows on canvas.
        Arrow length is proportional to speed; tiny vectors are skipped.
        """
        step   = self.arrow_grid
        scale  = 4.0        # pixels-per-unit of velocity
        min_mag = 0.15      # skip arrows below this magnitude

        h, w = flow.shape[:2]
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                vx = float(flow[y, x, 0])
                vy = float(flow[y, x, 1])
                mag = (vx**2 + vy**2) ** 0.5
                if mag < min_mag:
                    continue

                ex = int(x + vx * scale)
                ey = int(y + vy * scale)
                ex = max(0, min(w - 1, ex))
                ey = max(0, min(h - 1, ey))

                # Color by direction (same hue wheel as HSV overlay)
                ang_deg = int((np.degrees(np.arctan2(vy, vx)) % 360) / 2)
                hsv_px  = np.array([[[ang_deg, 255, 220]]], np.uint8)
                arrow_color = tuple(int(c) for c in cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0])

                cv2.arrowedLine(canvas, (x, y), (ex, ey),
                                arrow_color, 1, cv2.LINE_AA, tipLength=0.35)

    # ------------------------------------------------------------------ composite
    def render_frame(self, frame_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Blend original frame with HSV overlay and arrow grid.
        """
        hsv_layer = self._flow_to_hsv(flow)

        # Mask: only paint where there is real motion (avoids uniform color smear
        # on static background from noise)
        vx, vy  = flow[..., 0], flow[..., 1]
        mag     = np.sqrt(vx**2 + vy**2)
        motion  = (mag > 0.15).astype(np.float32)
        # Larger blur → softer blob edges matching Farneback's smooth falloff
        motion  = cv2.GaussianBlur(motion, (31, 31), 8.0)
        mask3   = np.stack([motion, motion, motion], axis=2)

        # Blend HSV only where there is motion
        composite = frame_bgr.astype(np.float32)
        composite = composite * (1 - mask3 * self.hsv_alpha) + \
                    hsv_layer.astype(np.float32) * mask3 * self.hsv_alpha
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        self._draw_arrows(composite, flow)

        return composite

    # ------------------------------------------------------------------ run
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        ret, frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.cvtColor(
            cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2GRAY
        )
        frame_counter = 0
        print("Running dense optical flow — press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (self.width, self.height))
            curr_gray     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            frame_counter += 1

            # Compute dense LK flow on full grid
            flow = self.compute_dense_flow(prev_gray, curr_gray)

            # Build visualisation
            output = self.render_frame(frame_resized, flow)

            # HUD
            vx, vy  = flow[..., 0], flow[..., 1]
            mag     = np.sqrt(vx**2 + vy**2)
            cv2.putText(output,
                        f"Frame {frame_counter} | Peak flow: {mag.max():.2f} px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Dense Optical Flow", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            prev_gray = curr_gray

        cap.release()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    app = DenseOpticalFlowManual(
        video_path  = "/home/CJ.mp4",
        width       = 832,
        height      = 464,
        grid_step   = 8,     # LK computed every 8px → denser, finer detail
        window_size = 21,    # Larger LK window → smoother per-point estimates
        arrow_grid  = 32,    # Arrow drawn every 32px
        hsv_alpha   = 0.6,   # HSV overlay strength
        arrow_alpha = 0.9,   # Arrow visibility
    )
    app.run()