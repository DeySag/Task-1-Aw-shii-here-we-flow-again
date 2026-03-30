"""
optical_flow_controller.py

Full implementation of:
  "Optical Flow based Visual Potential Field for Autonomous Driving"
  Capito, Ozguner, Redmill — IV 2020

Pipeline (per frame pair It, It+1):
  1.  Capture RGB frames from PyBullet windshield camera
  2.  Detect Shi-Tomasi corners on It  (cv2 allowed for detection)
  3.  Sparse LK optical flow — our hand-rolled LucasKanadeTracker
  4.  Focus of Expansion (FOE) via least-squares  (Eq. 1-2, Sec. III-A)
  5.  TTC per tracked point  (Eq. 3,  Sec. III-B)
  6.  Obstacle map Ot via Otsu → Gaussian G*Ot → gradient g(x,y,t) (Eq. 4-6)
  7.  Obstacle repulsive force F_rep  (Eq. 9,  Sec. IV-B)
  8.  Attractive force F_att          (Eq. 7-8, Sec. IV-A)
  9.  Road potential field Us          (Eq. 10-17, Sec. IV-C)
      → Road repulsive force F_reprd  (Eq. 18)
  10. Total force in global coords    (Eq. 19-21, Sec. V)
  11. Desired heading ψ_d             (Eq. 27,    Sec. VI)
  12. Gradient-Tracking Sliding Mode Controller:
        Lateral  — rotational manifold s_r  (Eq. 28-30)
        Longitudinal — s_l                 (Eq. 31-32)

Usage:
    python optical_flow_controller.py
"""

import cv2
import numpy as np
import pybullet as p
import time
import math

from simulation_setup import setup_simulation
from lucaskanade import LucasKanadeTracker

# ─────────────────────────────────────────────────────────────────────────────
# TUNEABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Image / camera
IMG_W      = 640
IMG_H      = 480
FOV_DEG    = 60
NEAR_CLIP  = 0.05
FAR_CLIP   = 20.0

# LK sparse flow
LK_LEVELS  = 3      # pyramid levels  (paper uses 3)
LK_WIN     = 15     # window size     (paper uses 25; 15 suits our resolution)
MAX_CORNERS = 150
QUAL_LVL    = 0.01
MIN_DIST    = 10    # minimum pixel distance between Shi-Tomasi features

# APF gains
ALPHA      = 0.6    # attractive gain  α
GAMMA      = 2.5    # obstacle repulsive gain  γ
LAMBDA_X   = 0.4    # road-force weight in X  λX
LAMBDA_Y   = 0.4    # road-force weight in Y  λY

# Road Morse potential (Table I)
APF_A        = 0.5
APF_B        = 1.0
C2_STRAIGHT  = 0.005
C2_CURVED    = 5e-6
C1           = 0.0
DELTA_X      = 1.0e-10   # small offset to avoid /0 at x=0  (Eq. 16)

# Road geometry — simulation lane markings at ±0.85 m
Y_RIGHT      =  0.85   # right lane boundary yr  [m]
Y_LEFT       = -0.85   # left  lane boundary yl  [m]

# GTSMC gains (Sec. VI)
CR           = 3.0    # rotational manifold gain
CL           = 1.0    # longitudinal manifold gain
U0           = 0.8    # max steering rate  [rad/s]
A0           = 0.4    # max acceleration magnitude  [m/s²]
V_DES        = 3.0    # desired speed  [m/s]  (~11 km/h)

# Steering smoothing (Fix 1 & 4)
S_R_SAT        = 0.3            # manifold saturation threshold [rad] — proportional zone
MAX_DELTA_DOT  = math.radians(30)  # max steering rate [rad/s] — prevents oscillation

# Road force centering bias gain (Fix 3)
Y_CENTER_GAIN  = 0.3            # proportional pull back to y=0

# FOE low-pass filter coefficient (Fix 2)
FOE_ALPHA      = 0.05           # small → slow filter, ignores transient controller turns

# Physics
DT           = 1.0 / 60.0
MAX_STEER    = math.radians(40)   # ±40° steering limit
MOTOR_FORCE  = 800

# Goal (end of slalom road)
X_GOAL = 31.0
Y_GOAL =  0.0


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE OPTICAL FLOW ENGINE  (wraps our LucasKanadeTracker)
# ─────────────────────────────────────────────────────────────────────────────

class SparseFlowEngine:
    """
    Produces a list of (tracked-point, flow-vector) pairs from two consecutive
    grayscale frames.  Feature detection uses cv2.goodFeaturesToTrack (Shi-Tomasi,
    which is the algorithm specified in the paper Sec. III).  The actual optical
    flow computation uses our hand-rolled LucasKanadeTracker exclusively.
    """

    def __init__(self):
        self.tracker = LucasKanadeTracker(
            num_pyramid_levels=LK_LEVELS,
            num_iterations=5
        )

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        """
        Shi-Tomasi corner detection.
        Returns (N, 2) array of [col, row] float positions, or empty (0,2).
        """
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUAL_LVL,
            minDistance=MIN_DIST
        )
        if pts is None:
            return np.empty((0, 2), dtype=np.float32)
        return pts.reshape(-1, 2)   # (N, [col, row])

    def compute(self, gray1: np.ndarray, gray2: np.ndarray):
        """
        Compute sparse flow It → It+1.

        Parameters
        ----------
        gray1, gray2 : uint8 grayscale images (H × W)

        Returns
        -------
        pts   : (N, 2) float  pixel positions (col=x, row=y) of tracked features
        flows : (N, 2) float  (vx, vy) optical flow vector per feature
        valid : (N,)   bool   True for points that produced a usable flow vector
        """
        f1 = gray1.astype(np.float64)
        f2 = gray2.astype(np.float64)

        pts = self._detect(gray1)
        if len(pts) == 0:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.empty(0, dtype=bool))

        flows = np.zeros_like(pts)
        valid = np.zeros(len(pts), dtype=bool)

        half = LK_WIN // 2
        h, w = gray1.shape

        for i, (col, row) in enumerate(pts):
            # Anchor the window top-left corner, clamped inside the image
            r = int(np.clip(int(row) - half, 0, h - LK_WIN - 1))
            c = int(np.clip(int(col) - half, 0, w - LK_WIN - 1))

            # Our pyramidal LK returns a (2,1) velocity [vx; vy]
            vel = self.tracker.pyramidal_lucas_kanade(f1, f2, r, c, LK_WIN)

            vx = float(vel[0, 0])
            vy = float(vel[1, 0])
            mag = math.hypot(vx, vy)

            # Accept only vectors with sensible magnitude
            if 1e-3 < mag < 80.0:
                flows[i] = [vx, vy]
                valid[i] = True

        return pts, flows, valid


# ─────────────────────────────────────────────────────────────────────────────
# SEC. III-A  —  FOCUS OF EXPANSION  (Eq. 1-2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_foe(pts: np.ndarray, flows: np.ndarray, valid: np.ndarray):
    """
    Least-squares FOE.

    For each tracked point p_i(x, y) with flow (vx, vy):
        a_i0 = vy,    a_i1 = -vx,    b_i = x*vy - y*vx    (Eq. 1-2)

    FOE = (A^T A)^{-1} A^T b

    Returns (xFOE, yFOE) in image-plane pixel coords.
    Falls back to image centre when fewer than 4 valid points.
    """
    idx = np.where(valid)[0]
    if len(idx) < 4:
        return float(IMG_W / 2), float(IMG_H / 2)

    x  = pts[idx, 0].astype(np.float64)   # image-plane x (col)
    y  = pts[idx, 1].astype(np.float64)   # image-plane y (row)
    vx = flows[idx, 0].astype(np.float64)
    vy = flows[idx, 1].astype(np.float64)

    a0 =  vy          # a_i0 column
    a1 = -vx          # a_i1 column
    b  = x * vy - y * vx

    A_mat = np.stack([a0, a1], axis=1)   # (N, 2)
    ATA   = A_mat.T @ A_mat              # (2, 2)
    ATb   = A_mat.T @ b                  # (2,)

    try:
        foe = np.linalg.solve(ATA, ATb)
        xf  = float(np.clip(foe[0], -IMG_W, 2 * IMG_W))
        yf  = float(np.clip(foe[1], -IMG_H, 2 * IMG_H))
        return xf, yf
    except np.linalg.LinAlgError:
        return float(IMG_W / 2), float(IMG_H / 2)


# ─────────────────────────────────────────────────────────────────────────────
# SEC. III-B  —  TIME TO CONTACT  (Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ttc(pts, flows, valid, xFOE, yFOE):
    """
    TTC_i = sqrt((x - xFOE)^2 + (y - yFOE)^2) / sqrt(vx^2 + vy^2)   Eq.(3)

    Returns (N,) array; np.inf for invalid / stationary points.
    """
    ttc = np.full(len(pts), np.inf)
    for i in range(len(pts)):
        if not valid[i]:
            continue
        vmag = math.hypot(flows[i, 0], flows[i, 1])
        if vmag > 1e-6:
            dist     = math.hypot(pts[i, 0] - xFOE, pts[i, 1] - yFOE)
            ttc[i]   = dist / vmag
    return ttc


# ─────────────────────────────────────────────────────────────────────────────
# SEC. III-C  —  OBSTACLE DETECTION  →  gradient field g(x,y,t)  (Eq. 4-6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_obstacle_gradient(pts, flows, valid, img_shape):
    """
    Steps
    -----
    1. Build a dense flow-magnitude image M from sparse tracked points.
    2. Otsu threshold → binary obstacle map O_t(x,y)  (Sec. III-C).
    3. Convolve with Gaussian G  (Eq. 4-5);  σ = IMG_W / 2  per paper.
    4. ∇(G * O)  via Sobel  →  g(x,y,t)  (Eq. 6).

    Returns
    -------
    gx, gy : (H, W) float64  — x and y components of obstacle gradient
    O_bin  : (H, W) uint8    — binary obstacle mask (for visualisation)
    """
    H, W = img_shape

    # Step 1 : scatter flow magnitudes onto a canvas
    M = np.zeros((H, W), dtype=np.float32)
    for i in range(len(pts)):
        if not valid[i]:
            continue
        cx  = int(np.clip(pts[i, 0], 0, W - 1))
        cy  = int(np.clip(pts[i, 1], 0, H - 1))
        mag = math.hypot(flows[i, 0], flows[i, 1])
        if mag > M[cy, cx]:
            M[cy, cx] = mag

    # Dilate to cover the sparse point cloud
    M = cv2.dilate(M, np.ones((7, 7), np.uint8))

    # Step 2 : Otsu threshold on flow-magnitude image
    M_u8 = np.clip(M * 10.0, 0, 255).astype(np.uint8)
    _, O_bin = cv2.threshold(M_u8, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3 : Gaussian smoothing  (Eq. 4-5)
    # Paper sets σ = half image width.  We cap the kernel to a practical size.
    sigma  = W / 2.0
    ksize  = int(6 * sigma) | 1          # round up to odd integer
    ksize  = min(ksize, (min(H, W) - 1) | 1)
    G_O    = cv2.GaussianBlur(O_bin.astype(np.float32), (ksize, ksize), sigma)

    # Step 4 : gradient  ∇(G*O)  (Eq. 6)
    gx = cv2.Sobel(G_O, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(G_O, cv2.CV_64F, 0, 1, ksize=5)

    return gx, gy, O_bin


# ─────────────────────────────────────────────────────────────────────────────
# SEC. IV-A  —  ATTRACTIVE FORCE  (Eq. 7-8)
# ─────────────────────────────────────────────────────────────────────────────

def compute_attractive_force(car_x, car_y, goal_x=X_GOAL, goal_y=Y_GOAL):
    """
    U_att = ½ α ‖p - p_goal‖                         Eq.(7)
    F_att = ∇U_att = α * (p_goal - p) / ‖p_goal - p‖ Eq.(8)

    Returns (Fattx, Fatty) in motion-plane (X = forward, Y = lateral).
    """
    dx   = goal_x - car_x
    dy   = goal_y - car_y
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return 0.0, 0.0
    return ALPHA * dx / dist, ALPHA * dy / dist


# ─────────────────────────────────────────────────────────────────────────────
# SEC. IV-B  —  OBSTACLE REPULSIVE FORCE  (Eq. 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_obstacle_repulsive_force(gx, gy, pts, ttc, valid):
    """
    F_rep = (γ / |R|) * (Σ_A g(p_i)) / (Σ_A TTC_i)     Eq.(9)

    |R| = total pixel count in region of interest.

    Returns (Fx, Fy).
    """
    H, W   = gx.shape
    R      = float(H * W)

    sum_gx  = 0.0
    sum_gy  = 0.0
    sum_ttc = 0.0

    for i in range(len(pts)):
        if not valid[i]:
            continue
        cx = int(np.clip(pts[i, 0], 0, W - 1))
        cy = int(np.clip(pts[i, 1], 0, H - 1))
        sum_gx  += gx[cy, cx]
        sum_gy  += gy[cy, cx]
        t        = ttc[i]
        if np.isfinite(t):
            sum_ttc += t

    if abs(sum_ttc) < 1e-9:
        return 0.0, 0.0

    Fx = GAMMA / R * (sum_gx / sum_ttc)
    Fy = GAMMA / R * (sum_gy / sum_ttc)
    return Fx, Fy


# ─────────────────────────────────────────────────────────────────────────────
# SEC. IV-C  —  ROAD POTENTIAL FIELD  (Eq. 10-18)
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_positions(car_x, c2):
    """
    Compute yr(x) and yl(x) from Eq.(14-15).

    yr = c2*(x+δx)^n + c1*(x+δx) + c0r
    yl = c2*(x+δx)^n + c1*(x+δx) + c0l

    n = 1 for straight, n = 2 for curved (not used here; straight track).
    """
    xd = car_x + DELTA_X
    yr = c2 * xd + C1 * xd + Y_RIGHT
    yl = c2 * xd + C1 * xd + Y_LEFT
    return yr, yl


def _morse_pair(car_y, yr, yl, car_x, c2):
    """
    Evaluate U_sr (Eq. 12) and U_sl (Eq. 13) at the given (car_x, car_y).

    U_sr = A * (1 - exp(-b*sign(y-yr) * sqrt( ((y-by_r)/m - (x+δx))^2 + (yr-y)^2 )))^2
    U_sl = A * (1 - exp( b*sign(y-yl) * sqrt( ((y-by_l)/m - (x+δx))^2 + (yl-y)^2 )))^2
    """
    xd = car_x + DELTA_X

    # Slope of line perpendicular to lane centre  Eq.(16)
    denom = 2.0 * c2 * xd + C1
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    m = -1.0 / denom

    # y-intercept of that perpendicular line  Eq.(17)
    by_r = yr - m * xd
    by_l = yl - m * xd

    # ---- Right boundary  Eq.(12) ----
    inner_r = math.sqrt(((car_y - by_r) / m - xd) ** 2 + (yr - car_y) ** 2)
    sign_r  = math.copysign(1.0, car_y - yr)
    U_sr    = APF_A * (1.0 - math.exp(-APF_B * sign_r * inner_r)) ** 2

    # ---- Left boundary  Eq.(13) ----
    inner_l = math.sqrt(((car_y - by_l) / m - xd) ** 2 + (yl - car_y) ** 2)
    sign_l  = math.copysign(1.0, car_y - yl)
    U_sl    = APF_A * (1.0 - math.exp( APF_B * sign_l * inner_l)) ** 2

    return U_sr, U_sl


# Module-level state for the FOE low-pass filter (Fix 2).
# Keeps the road curvature estimate stable across frames.
_foe_x_filtered = float(IMG_W / 2)


def compute_road_repulsive_force(car_x, car_y, xFOE, img_cx=IMG_W / 2.0):
    """
    1. Low-pass filter xFOE so transient controller turns don't flip the
       road curvature model (Fix 2).
    2. Choose c2 from filtered FOE offset (straight vs curved, Sec. IV-C).
    3. Evaluate Us = U_sr + U_sl  (Eq. 10-13).
    4. Numerical gradient  F_reprd = ∇Us  (Eq. 18).
    5. Add centering bias — gentle proportional pull toward y=0 (Fix 3).

    Returns (FX_road, FY_road) in motion-plane coords.
    """
    global _foe_x_filtered

    # Fix 2: exponential moving average on FOE x position.
    # Only long-lived FOE shifts (genuine curves) affect c2.
    _foe_x_filtered = (1.0 - FOE_ALPHA) * _foe_x_filtered + FOE_ALPHA * xFOE

    foe_offset = _foe_x_filtered - img_cx
    if abs(foe_offset) < 0.10 * IMG_W:
        c2 = C2_STRAIGHT
    elif foe_offset > 0:
        c2 = C2_CURVED       # right curve
    else:
        c2 = -C2_CURVED      # left curve

    yr,  yl  = _boundary_positions(car_x, c2)
    U_sr, U_sl = _morse_pair(car_y, yr, yl, car_x, c2)
    Us = U_sr + U_sl

    eps = 1e-4

    # ∂Us/∂y
    yr_dy, yl_dy = _boundary_positions(car_x, c2)
    U_sr_dy, U_sl_dy = _morse_pair(car_y + eps, yr_dy, yl_dy, car_x, c2)
    FY_road = -((U_sr_dy + U_sl_dy) - Us) / eps

    # ∂Us/∂x
    yr_dx, yl_dx = _boundary_positions(car_x + eps, c2)
    U_sr_dx, U_sl_dx = _morse_pair(car_y, yr_dx, yl_dx, car_x + eps, c2)
    FX_road = -((U_sr_dx + U_sl_dx) - Us) / eps

    # Fix 3: centering bias — always nudge toward y=0 proportionally.
    # This fills the near-zero gradient zone at lane centre and prevents
    # the car drifting and staying off-centre after an obstacle avoidance.
    FY_road += Y_CENTER_GAIN * (-car_y)

    return FX_road, FY_road


# ─────────────────────────────────────────────────────────────────────────────
# SEC. V  —  TOTAL POTENTIAL FIELD  (Eq. 19-21)
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_force(Fattx, Fatty,
                        Frepx, Frepy,
                        FX_road, FY_road,
                        psi):
    """
    Vehicle (motion-plane) frame  Eq.(19-20):
        FXT = Fattx - Frepx - λX * FX_road
        FYT = Fatty - Frepy - λY * FY_road

    Global frame  Eq.(21):
        [FX0]   [ cosψ  sinψ ] [FXT]
        [FY0] = [-sinψ  cosψ ] [FYT]

    Returns (FX0, FY0).
    """
    FXT = Fattx - Frepx - LAMBDA_X * FX_road
    FYT = Fatty - Frepy - LAMBDA_Y * FY_road

    cp = math.cos(psi)
    sp = math.sin(psi)
    FX0 =  cp * FXT + sp * FYT
    FY0 = -sp * FXT + cp * FYT
    return FX0, FY0


# ─────────────────────────────────────────────────────────────────────────────
# SEC. VI  —  GRADIENT-TRACKING SLIDING MODE CONTROLLER  (Eq. 27-32)
# ─────────────────────────────────────────────────────────────────────────────

class GTSMC:
    """
    Gradient-Tracking Sliding Mode Controller (Sec. VI of paper).

    Fixes applied vs. original bang-bang formulation:
      Fix 1 — Proportional-saturating steering rate replaces hard sign().
               Proportional within ±S_R_SAT, saturates outside → smooth
               small corrections, still decisive for large errors.
      Fix 4 — Per-frame steering rate clamped to MAX_DELTA_DOT so rapid
               force-vector jumps cannot slam the wheel to full lock.

    Lateral:      rotational manifold s_r  (Eq. 28-30)
    Longitudinal: manifold s_l             (Eq. 31-32)
    """

    def __init__(self):
        self.delta_f   = 0.0   # current steering angle  [rad]
        self.psi_prev  = 0.0   # heading from previous step (for ψ̇_e)

    def step(self, FX0: float, FY0: float,
             psi: float, speed: float, dt: float):
        """
        Parameters
        ----------
        FX0, FY0 : total force components in global frame
        psi      : current yaw  [rad]
        speed    : current speed  [m/s]
        dt       : time-step  [s]

        Returns
        -------
        delta_f : steering angle  [rad], clamped to ±MAX_STEER
        accel   : normalised acceleration command ∈ [-1, 1]
        """
        # ── Desired heading  Eq.(27) ──────────────────────────────────────
        if abs(FX0) < 1e-9 and abs(FY0) < 1e-9:
            psi_d = psi               # no force → hold heading
        else:
            psi_d = math.atan2(FY0, FX0)

        # ── Heading error & its derivative ───────────────────────────────
        psi_e     = psi - psi_d
        psi_e     = (psi_e + math.pi) % (2.0 * math.pi) - math.pi  # wrap to [-π, π]
        psi_dot_e = (psi - self.psi_prev) / dt
        self.psi_prev = psi

        # ── Rotational manifold  Eq.(28) ─────────────────────────────────
        s_r = CR * psi_e + psi_dot_e

        # ── Fix 1: proportional-saturating steering rate ──────────────────
        # Original: u = -U0 * sign(s_r)  — bang-bang, causes oscillation.
        # New:      u = -U0 * clip(s_r / S_R_SAT, -1, 1)
        #   • |s_r| < S_R_SAT  → proportional, rate proportional to error
        #   • |s_r| ≥ S_R_SAT  → saturated at ±U0, same as original but
        #                         only when error is genuinely large
        u_raw = -U0 * max(-1.0, min(1.0, s_r / S_R_SAT))

        # ── Fix 4: rate-limit how fast the wheel can turn ────────────────
        # Prevents force-vector discontinuities from slamming to full lock.
        max_change   = MAX_DELTA_DOT * dt
        u_limited    = max(-max_change, min(max_change, u_raw * dt))
        self.delta_f = float(np.clip(self.delta_f + u_limited,
                                     -MAX_STEER, MAX_STEER))

        # ── Longitudinal manifold  Eq.(31) ───────────────────────────────
        s_l   = CL * speed - V_DES

        # ── Throttle/brake  Eq.(32):  a = -a0 · sign(s_l) ───────────────
        accel = float(np.clip(-A0 * math.copysign(1.0, s_l), -1.0, 1.0))

        return self.delta_f, accel


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_camera_image(car_id):
    """
    Render a windshield-mounted front camera from PyBullet.

    The camera is positioned 0.5 m forward and 0.4 m above the car base,
    looking 3 m ahead and slightly downward so the road is visible.

    Returns
    -------
    rgb  : (H, W, 3) uint8  — RGB image
    gray : (H, W)    uint8  — greyscale image
    """
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot      = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    # Camera position: 0.5 m ahead, 0.4 m up from car base in local frame
    cam_local  = np.array([0.5, 0.0, 0.4])
    cam_pos    = np.array(pos) + rot @ cam_local

    # Look 3 m forward, 0.2 m down
    look_local  = np.array([3.0, 0.0, -0.2])
    look_target = np.array(pos) + rot @ look_local
    up_vec      = (rot @ np.array([0.0, 0.0, 1.0])).tolist()

    view_mat = p.computeViewMatrix(
        cameraEyePosition    = cam_pos.tolist(),
        cameraTargetPosition = look_target.tolist(),
        cameraUpVector       = up_vec
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov     = FOV_DEG,
        aspect  = IMG_W / IMG_H,
        nearVal = NEAR_CLIP,
        farVal  = FAR_CLIP
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width            = IMG_W,
        height           = IMG_H,
        viewMatrix       = view_mat,
        projectionMatrix = proj_mat,
        renderer         = p.ER_TINY_RENDERER
    )

    rgb  = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


# ─────────────────────────────────────────────────────────────────────────────
# PYBULLET CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def apply_controls(car_id, steering_joints, motor_joints, delta_f, accel):
    """
    Map scalar steering angle and normalised acceleration to PyBullet joints.
    """
    for j in steering_joints:
        p.setJointMotorControl2(
            car_id, j,
            p.POSITION_CONTROL,
            targetPosition = float(delta_f),
            force          = 10
        )
    target_vel = accel * 20.0        # m/s → rad/s (approximate wheel scaling)
    for j in motor_joints:
        p.setJointMotorControl2(
            car_id, j,
            p.VELOCITY_CONTROL,
            targetVelocity = target_vel,
            force          = MOTOR_FORCE
        )


def get_car_state(car_id):
    """
    Returns (x, y, yaw [rad], speed [m/s]).
    """
    pos, orn  = p.getBasePositionAndOrientation(car_id)
    vel, _    = p.getBaseVelocity(car_id)
    yaw       = p.getEulerFromQuaternion(orn)[2]
    speed     = math.hypot(vel[0], vel[1])
    return pos[0], pos[1], yaw, speed


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_debug(rgb, pts, flows, valid, xFOE, yFOE, O_bin,
               delta_f, speed, car_x, car_y):
    """
    Overlay optical flow arrows, FOE dot, obstacle mask, and HUD text
    onto a copy of the camera frame.
    """
    vis = rgb.copy()

    # Obstacle mask — semi-transparent red tint
    tint = np.zeros_like(vis)
    tint[O_bin > 0] = [220, 60, 60]
    vis = cv2.addWeighted(vis, 0.80, tint, 0.20, 0)

    # Flow vectors — green arrows
    for i in range(len(pts)):
        if not valid[i]:
            continue
        x0, y0 = int(pts[i, 0]), int(pts[i, 1])
        x1 = int(x0 + flows[i, 0] * 4)
        y1 = int(y0 + flows[i, 1] * 4)
        cv2.arrowedLine(vis, (x0, y0), (x1, y1),
                        (0, 230, 0), 1, tipLength=0.35)

    # FOE — blue filled circle
    fx = int(np.clip(xFOE, 2, IMG_W - 3))
    fy = int(np.clip(yFOE, 2, IMG_H - 3))
    cv2.circle(vis, (fx, fy), 9, (255, 60, 0), -1)
    cv2.putText(vis, "FOE", (fx + 11, fy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 0), 1)

    # HUD — yellow text
    hud_lines = [
        f"X  : {car_x:6.2f} m",
        f"Y  : {car_y:6.3f} m",
        f"v  : {speed:5.2f} m/s",
        f"d  : {math.degrees(delta_f):+5.1f} deg",
        f"pts: {int(valid.sum()):3d}",
        f"FOE: ({xFOE:.0f}, {yFOE:.0f})",
    ]
    for j, txt in enumerate(hud_lines):
        cv2.putText(vis, txt, (8, 20 + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    # ── Initialise simulation ────────────────────────────────────────────────
    car_id, steering_joints, motor_joints = setup_simulation(
        dt=DT, settle_frames=60, gui=True
    )

    flow_engine = SparseFlowEngine()
    controller  = GTSMC()

    # Grab first frame so we have a "previous" on the very first real step
    _, gray_prev = get_camera_image(car_id)
    p.stepSimulation()
    time.sleep(DT)

    frame_idx = 0
    print("\n[Controller] Running — press Q in the OpenCV window to stop.\n")

    while True:
        frame_idx += 1

        # 1. Camera frame ─────────────────────────────────────────────────────
        rgb, gray_curr = get_camera_image(car_id)

        # 2. Sparse optical flow  (our LucasKanadeTracker) ────────────────────
        pts, flows, valid = flow_engine.compute(gray_prev, gray_curr)

        # 3. FOE  (Eq. 1-2) ───────────────────────────────────────────────────
        xFOE, yFOE = compute_foe(pts, flows, valid)

        # 4. TTC  (Eq. 3) ─────────────────────────────────────────────────────
        ttc = compute_ttc(pts, flows, valid, xFOE, yFOE)

        # 5. Obstacle gradient  (Eq. 4-6) ─────────────────────────────────────
        gx, gy, O_bin = compute_obstacle_gradient(
            pts, flows, valid, (IMG_H, IMG_W)
        )

        # 6. Obstacle repulsive force  (Eq. 9) ────────────────────────────────
        Frepx, Frepy = compute_obstacle_repulsive_force(gx, gy, pts, ttc, valid)

        # 7. Car state ────────────────────────────────────────────────────────
        car_x, car_y, psi, speed = get_car_state(car_id)

        # 8. Attractive force  (Eq. 7-8) ──────────────────────────────────────
        Fattx, Fatty = compute_attractive_force(car_x, car_y)

        # 9. Road repulsive force  (Eq. 10-18) ────────────────────────────────
        FX_road, FY_road = compute_road_repulsive_force(
            car_x, car_y, xFOE, IMG_W / 2.0
        )

        # 10. Total force → global frame  (Eq. 19-21) ─────────────────────────
        FX0, FY0 = compute_total_force(
            Fattx, Fatty,
            Frepx, Frepy,
            FX_road, FY_road,
            psi
        )

        # 11. GTSMC steering + throttle  (Eq. 27-32) ──────────────────────────
        delta_f, accel = controller.step(FX0, FY0, psi, speed, DT)

        # 12. Apply to simulation ─────────────────────────────────────────────
        apply_controls(car_id, steering_joints, motor_joints, delta_f, accel)

        # 13. Step physics ────────────────────────────────────────────────────
        p.stepSimulation()
        time.sleep(DT)

        # 14. Visualise ───────────────────────────────────────────────────────
        vis = draw_debug(rgb, pts, flows, valid,
                         xFOE, yFOE, O_bin,
                         delta_f, speed, car_x, car_y)
        cv2.imshow("Optical Flow  |  Visual Potential Field", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Controller] User requested quit.")
            break

        # 15. Roll frames ─────────────────────────────────────────────────────
        gray_prev = gray_curr.copy()

        # 16. Goal reached? ───────────────────────────────────────────────────
        if math.hypot(X_GOAL - car_x, Y_GOAL - car_y) < 1.0:
            print(f"[Controller] Goal reached at frame {frame_idx}. Done.")
            break

        # 17. Periodic console log ────────────────────────────────────────────
        if frame_idx % 60 == 0:
            print(
                f"  frame={frame_idx:5d}  "
                f"x={car_x:6.2f}  y={car_y:+5.3f}  "
                f"ψ={math.degrees(psi):+5.1f}°  "
                f"v={speed:.2f} m/s  "
                f"δ={math.degrees(delta_f):+5.1f}°  "
                f"F=({FX0:+.3f}, {FY0:+.3f})  "
                f"FOE=({xFOE:.0f}, {yFOE:.0f})  "
                f"pts={int(valid.sum())}"
            )

    cv2.destroyAllWindows()
    try:
        p.disconnect()
    except Exception:
        pass
    print("[Done]")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
