"""
visual_potential_field_nav.py

Full implementation of:
    "Optical Flow based Visual Potential Field for Autonomous Driving"
    Capito, Ozguner, Redmill – IEEE Intelligent Vehicles Symposium, 2020

This version uses the author-supplied simulation_setup.py directly.
The SimulationEnvironment class is now a thin adapter around the
setup_simulation() helper provided by that module; all pipeline logic
(Sections III – VI) is provided.

Behaviour improvements over the base implementation
────────────────────────────────────────────────────
  FIX-1  Softer steering
         GTSMC: u0 1.0 → 0.40, delta_max 40° → 22°, cr raised to 3.5.
         EMA low-pass filter applied to both psi_d and delta_f so that
         noisy optical-flow frames do not cause abrupt wheel snaps.

  FIX-2  Hard boundary repulsion  (boundary_force)
         Exponential wall force that activates within 0.55 m of the road
         edges (±1.16 m) and grows steeply as the car approaches the wall.
         Added with its own gain lam_bnd in compute_desired_heading so it
         ALWAYS overrides obstacle or road forces near the edge.

  FIX-3  Lane-centre re-centring  (centering_force)
         Gentle proportional pull toward y = 0 whenever the car is off
         centre.  Deliberately weak (ctr_k = 0.25) so it does not fight
         active obstacle avoidance, but provides a clear restoring bias
         once the obstacle is cleared.

Pipeline (one frame):
    capture → Shi-Tomasi + Pyramidal LK  (Sec. III)
            → FOE via least-squares       (Sec. III-A, Eq. 1-3)
            → TTC per tracked point       (Sec. III-B, Eq. 3)
            → Obstacle map + gradient     (Sec. III-C, Eq. 4-6)
            → Attractive force            (Sec. IV-A,  Eq. 7-8)
            → Repulsive force             (Sec. IV-B,  Eq. 9)
            → Road potential field        (Sec. IV-C,  Eq. 10-18)
            → Boundary repulsion          (FIX-2)
            → Lane-centre pull            (FIX-3)
            → Total force → psi_d         (Sec. V,     Eq. 19-21, 27)
            → EMA-smoothed psi_d          (FIX-1)
            → GTSMC steering + speed      (Sec. VI,    Eq. 28-32)
            → EMA-smoothed delta_f        (FIX-1)
            → PyBullet racecar control    (Sec. VI-A,  Eq. 22-26)
"""

import math
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data

from lucaskanade import LucasKanadeTracker          # manual Pyramidal LK
from simulation_setup import setup_simulation        # ← author-supplied module


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SIMULATION ENVIRONMENT  (adapter around author's simulation_setup.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationEnvironment:
    """
    Thin adapter that wraps setup_simulation() from the author's module.

    setup_simulation() handles:
      • p.connect() / gravity / timestep
      • Flat road with lane markings (33.3 m long, ±1.16 m half-width)
      • Five yellow/black slalom obstacles at x = 6, 12, 18, 24 m
      • End wall at x ≈ 31.7 m
      • Racecar URDF (globalScaling = 1.8) at the origin

    This class adds:
      • Goal-sphere visual marker
      • Front-mounted monocular camera (640×480, FOV 60°)
      • get_state() / apply_bicycle_control() / step() / disconnect()

    All pipeline code in sections 2-10 is unchanged from the original.

    Road geometry (from simulation_setup.py):
      Road half-width  = 1.16 m
      Lane markings at y = 0, ±0.85 m
    The road potential-field offsets c0r / c0l in VisualPotentialField are
    set to ±0.85 m to align with these markings (see Section 6 below).
    """

    CAM_W    = 640
    CAM_H    = 480
    CAM_FOV  = 60.0
    CAM_NEAR = 0.1
    CAM_FAR  = 100.0

    # Racecar wheel radius:  stock ~0.1 m × globalScaling 1.8 = 0.18 m
    WHEEL_RADIUS = 0.18   # (m)  used to convert m/s → rad/s

    def __init__(self, gui: bool = True):
        # ── Author's scene initialisation 
        # setup_simulation() calls p.connect(), sets gravity/timestep,
        # loads the plane, builds the road, spawns the racecar, and runs
        # 60 settle frames so suspension reaches equilibrium before we start.
        self.car_id, self.steering_joints, self.motor_joints = \
            setup_simulation(dt=1.0 / 60.0, settle_frames=60, gui=gui)

        # ── Camera debug view
        p.resetDebugVisualizerCamera(
            cameraDistance=8, cameraPitch=-20, cameraYaw=180,
            cameraTargetPosition=[5, 0, 0]
        )

        # ── Goal marker (green sphere near end wall)
        self.goal_world = np.array([29.0, 0.0, 0.35])
        gv = p.createVisualShape(p.GEOM_SPHERE, radius=0.5,
                                 rgbaColor=[0.1, 0.9, 0.1, 0.8])
        p.createMultiBody(0, -1, gv, self.goal_world.tolist())

        # ── Lane boundary references (world y, used by road potential field)
        self.lane_left_y  =  1.16   # left  road edge (m)
        self.lane_right_y = -1.16   # right road edge (m)

    # ── Camera
    def capture_frame(self):
        """
        Returns (bgr_frame, view_matrix, proj_matrix).
        Camera eye is 0.7 m ahead of and 0.3 m above the vehicle centre.
        """
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        eye    = np.array(pos) + R @ np.array([0.7, 0.0, 0.3])
        target = eye + R @ np.array([1.0, 0.0, 0.0])

        vm = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        pm = p.computeProjectionMatrixFOV(
            self.CAM_FOV, self.CAM_W / self.CAM_H,
            self.CAM_NEAR, self.CAM_FAR
        )
        _, _, rgba, _, _ = p.getCameraImage(
            self.CAM_W, self.CAM_H, vm, pm,
            renderer=p.ER_TINY_RENDERER
        )
        bgr = cv2.cvtColor(
            np.array(rgba, np.uint8).reshape(self.CAM_H, self.CAM_W, 4)[:, :, :3],
            cv2.COLOR_RGB2BGR
        )
        return bgr, vm, pm

    def project_to_image(self, world_pt, view_matrix, proj_matrix):
        """
        Project a 3-D world point to (u, v) pixel coords using the camera MVP.
        Returns None if the point is behind the camera or outside the frame.
        """
        vm = np.array(view_matrix).reshape(4, 4).T
        pm = np.array(proj_matrix).reshape(4, 4).T
        ph = np.array([*world_pt, 1.0], np.float64)
        clip = pm @ vm @ ph
        if abs(clip[3]) < 1e-8 or clip[2] < 0:
            return None
        ndc = clip[:3] / clip[3]
        u = int((ndc[0] + 1.0) * 0.5 * self.CAM_W)
        v = int((1.0 - ndc[1]) * 0.5 * self.CAM_H)
        if 0 <= u < self.CAM_W and 0 <= v < self.CAM_H:
            return (u, v)
        return None

    # ── Vehicle state
    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        vel, _   = p.getBaseVelocity(self.car_id)
        yaw      = p.getEulerFromQuaternion(orn)[2]
        return {
            'pos'  : np.array(pos),
            'yaw'  : yaw,
            'vel'  : np.array(vel),
            'speed': float(np.linalg.norm(vel[:2])),
        }

    # ── Bicycle kinematic model control (Eq. 22-26)
    def apply_bicycle_control(self, delta_f: float, speed: float):
        """
        Translates the bicycle-model outputs (Eq. 22-26) into PyBullet
        joint commands for the racecar URDF.

        Steering joints → POSITION_CONTROL with target angle delta_f.
        Motor joints    → VELOCITY_CONTROL  with target wheel angular
                          velocity  ω = speed / WHEEL_RADIUS  (rad/s).

        The racecar URDF exposes the same steering/motor joint lists that
        setup_simulation() already discovered and stored in
        self.steering_joints / self.motor_joints.
        """
        # Lateral (Eq. 25 → steer angle)
        for j in self.steering_joints:
            p.setJointMotorControl2(
                self.car_id, j,
                p.POSITION_CONTROL,
                targetPosition=delta_f,
                force=10.0
            )

        # Longitudinal (Eq. 26 → wheel speed)
        wheel_vel = speed / self.WHEEL_RADIUS     # m/s → rad/s
        for j in self.motor_joints:
            p.setJointMotorControl2(
                self.car_id, j,
                p.VELOCITY_CONTROL,
                targetVelocity=wheel_vel,
                force=800.0
            )

    def step(self):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    def disconnect(self):
        p.disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SPARSE OPTICAL FLOW  (Section III)
#     Shi-Tomasi corners + our manual Pyramidal Lucas-Kanade tracker
# ═══════════════════════════════════════════════════════════════════════════════

class SparseOpticalFlow:
    """
    Implements the optical flow block described in Section III:
      1. Convert to grayscale
      2. Detect corners with Shi-Tomasi (cv2.goodFeaturesToTrack)
      3. Track with Pyramidal LK (our LucasKanadeTracker)
         Paper params: window=(25,25), 3 pyramid levels, eps=0.03
    """

    def __init__(self,
                 lk_win      = 25,
                 lk_levels   = 3,
                 max_corners = 200,
                 quality     = 0.01,
                 min_dist    = 10):
        self.lk          = LucasKanadeTracker(num_pyramid_levels=lk_levels,
                                               num_iterations=1)
        self.lk_win      = lk_win
        self.max_corners = max_corners
        self.quality     = quality
        self.min_dist    = min_dist
        self._prev_gray  = None
        self._prev_pts   = None   # ndarray (N, 2) float32 – [x, y] pixel positions

    # ── Corner detection
    def _detect(self, gray: np.ndarray) -> np.ndarray:
        c = cv2.goodFeaturesToTrack(gray,
                                    maxCorners=self.max_corners,
                                    qualityLevel=self.quality,
                                    minDistance=self.min_dist)
        return c.reshape(-1, 2).astype(np.float32) if c is not None \
               else np.empty((0, 2), np.float32)

    # ── Main compute
    def compute(self, gray: np.ndarray):
        """
        Feed the next grayscale frame.
        Returns list of (x, y, vx, vy) for every successfully tracked point.
        """
        if self._prev_gray is None or len(self._prev_pts) == 0:
            self._prev_gray = gray
            self._prev_pts  = self._detect(gray)
            return []

        h, w = gray.shape
        half = self.lk_win // 2
        flows, next_pts = [], []

        for (px, py) in self._prev_pts:
            col = int(np.clip(px - half, 0, w - self.lk_win - 1))
            row = int(np.clip(py - half, 0, h - self.lk_win - 1))

            vel = self.lk.pyramidal_lucas_kanade(
                self._prev_gray.astype(np.float64),
                gray.astype(np.float64),
                row, col, self.lk_win
            )
            if vel is None:
                continue

            vx, vy = float(vel[0, 0]), float(vel[1, 0])
            mag    = math.hypot(vx, vy)

            if mag < 0.05 or mag > 40.0:
                continue

            nx, ny = px + vx, py + vy
            if 0 <= nx < w and 0 <= ny < h:
                flows.append((float(px), float(py), vx, vy))
                next_pts.append([nx, ny])

        if len(next_pts) < 20:
            self._prev_pts = self._detect(gray)
        else:
            self._prev_pts = np.array(next_pts, np.float32)

        self._prev_gray = gray
        return flows   # [(x, y, vx, vy), ...]

    def reset(self):
        self._prev_gray = None
        self._prev_pts  = None


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FOCUS OF EXPANSION  (Section III-A, Eq. 1-3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_foe(flows, img_w: int, img_h: int):
    """
    FOE via least-squares.  Each flow vector (vx, vy) at pixel (x, y)
    defines the constraint that the FOE lies on the line through (x,y)
    in direction (vx, vy):

        vy * (xF - x) - vx * (yF - y) = 0
        ⟹  [vy  -vx] [xF  yF]^T  =  x*vy - y*vx

    Building A (rows = [a_i0, a_i1] = [vy_i, -vx_i]) and
              b (rows = b_i = x_i*vy_i - y_i*vx_i),
    the closed-form least-squares solution is (Eq. 1-3):

        FOE = (A^T A)^{-1} A^T b

    Expanded (paper's explicit form):
        D    = Σa₀² · Σa₁² - (Σa₀a₁)²
        xFOE = (Σa₀b · Σa₁² - Σa₁b · Σa₀a₁) / D
        yFOE = (−Σa₀b · Σa₀a₁ + Σa₁b · Σa₀²) / D
    """
    if len(flows) < 4:
        return img_w / 2.0, img_h / 2.0

    a0 = np.array([ f[3]                   for f in flows], np.float64)  # vy
    a1 = np.array([-f[2]                   for f in flows], np.float64)  # -vx
    b  = np.array([f[0]*f[3] - f[1]*f[2]  for f in flows], np.float64)  # x·vy − y·vx

    Sa0b = float(np.dot(a0, b))
    Sa1b = float(np.dot(a1, b))
    Sa02 = float(np.dot(a0, a0))
    Sa12 = float(np.dot(a1, a1))
    Sa01 = float(np.dot(a0, a1))

    D = Sa02 * Sa12 - Sa01 ** 2
    if abs(D) < 1e-9:
        return img_w / 2.0, img_h / 2.0

    foe_x = (Sa0b * Sa12 - Sa1b * Sa01) / D
    foe_y = (-Sa0b * Sa01 + Sa1b * Sa02) / D
    return float(foe_x), float(foe_y)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TIME TO CONTACT  (Section III-B, Eq. 3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ttc(flows, foe_x: float, foe_y: float):
    """
    TTC_i = √[(x−xFOE)²+(y−yFOE)²] / √[vx²+vy²]   (Eq. 3)

    Returns a list of TTC values in the same order as `flows`.
    Very slow flows get a large TTC (not a collision threat).
    """
    ttcs = []
    for (x, y, vx, vy) in flows:
        dist_foe = math.hypot(x - foe_x, y - foe_y)
        flow_mag = math.hypot(vx, vy)
        ttcs.append(dist_foe / flow_mag if flow_mag > 1e-4 else 1e6)
    return ttcs


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  OBSTACLE DETECTION & GRADIENT FIELD  (Section III-C, Eq. 4-6)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_obstacle_gradient(flows, img_h: int, img_w: int):
    """
    Step 1 – Build a flow-magnitude image at tracked pixel positions.
    Step 2 – Otsu threshold on that image → binary obstacle mask O(x,y,t).
    Step 3 – Convolve O with a wide Gaussian G(x,y) with σ = img_w/2  (Eq. 4-5).
    Step 4 – g(x,y,t) = ∇(G * O)  (Eq. 6), computed via Sobel on the blurred mask.

    Returns (g_x, g_y) gradient arrays, each (img_h, img_w) float32.
    """
    # --- Step 1: scatter flow magnitudes onto a sparse image
    mag_img = np.zeros((img_h, img_w), np.float32)
    for (x, y, vx, vy) in flows:
        ix = int(np.clip(x, 0, img_w - 1))
        iy = int(np.clip(y, 0, img_h - 1))
        mag_img[iy, ix] = max(mag_img[iy, ix], math.hypot(vx, vy))

    if mag_img.max() < 1e-4:
        return (np.zeros((img_h, img_w), np.float32),
                np.zeros((img_h, img_w), np.float32))

    # --- Step 2: Otsu threshold → obstacle binary mask O
    mag_u8 = cv2.normalize(mag_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, O_bin = cv2.threshold(mag_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    O = O_bin.astype(np.float32) / 255.0

    # --- Step 3: G * O  (σ = half image width, Eq. 4)
    sigma = img_w / 2.0
    ksize = int(6 * sigma) | 1
    ksize = min(ksize, min(img_w, img_h) - 1) | 1
    GO = cv2.GaussianBlur(O, (ksize, ksize), sigma)

    # --- Step 4: gradient of G*O  (Eq. 6) 
    g_x = cv2.Sobel(GO, cv2.CV_32F, 1, 0, ksize=3)
    g_y = cv2.Sobel(GO, cv2.CV_32F, 0, 1, ksize=3)
    return g_x, g_y


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUAL POTENTIAL FIELD  (Section IV)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualPotentialField:
    """
    Computes all three potential field force components:
      IV-A  Attractive force      Eq. 7-8
      IV-B  Obstacle repulsive    Eq. 9
      IV-C  Road repulsive        Eq. 10-18  (Morse potential)

    Road-layout note:
      The author's simulation_setup.py builds a single narrow road
      (half-width = 1.16 m, lane markings at y = ±0.85 m).  To align
      the Morse potential boundaries with the actual markings the offsets
      c0r / c0l are set to ±0.85 m here (paper Table I uses ±5.25/−8.75 m
      for a 14 m wide, four-lane CARLA highway).  All other parameters and
      all equation logic are unchanged from the paper.
    """

    def __init__(self,
                 alpha     = 0.06,   # attractive gain              (tunable)
                 gamma     = 3.0,    # obstacle repulsive gain      (tunable)
                 road_A    = 0.5,    # Morse depth                  (Table I)
                 road_b    = 1.0,    # Morse variance parameter     (Table I)
                 c0r       =  0.85,  # right lane marking (m)       ← adapted
                 c0l       = -0.85,  # left  lane marking (m)       ← adapted
                 c2_str    = 0.005,  # c2 straight road             (Table I)
                 c2_cur    = 5e-6,   # c2 curved road               (Table I)
                 # ── FIX-2 : hard boundary parameters ──────────────────────
                 bnd_k     = 8.0,    # peak exponential gain at the wall
                 bnd_sigma = 0.18,   # exponential decay distance (m)
                 bnd_onset = 0.55,   # distance from wall where force begins (m)
                 # ── FIX-3 : lane-centre restoring force ───────────────────
                 ctr_k     = 0.25):  # proportional gain toward y = 0
        self.alpha     = alpha
        self.gamma     = gamma
        self.road_A    = road_A
        self.road_b    = road_b
        self.c0r       = c0r
        self.c0l       = c0l
        self.c2_str    = c2_str
        self.c2_cur    = c2_cur
        self.bnd_k     = bnd_k
        self.bnd_sigma = bnd_sigma
        self.bnd_onset = bnd_onset
        self.ctr_k     = ctr_k

    # ── IV-A : Attractive force  (Eq. 7-8)
    def attractive_force(self, goal_img, img_w: int, img_h: int) -> np.ndarray:
        """
        The target potential is proportional to Euclidean distance from the
        image centre to the projected goal pixel (Eq. 7).

        F_att = α · dist,   directed toward goal  (Eq. 8)
        θ_goal = atan2(y_goal − y_ctr, x_goal − x_ctr)

        If the goal is off-screen we return zero lateral bias (drive straight).
        """
        cx, cy = img_w / 2.0, img_h / 2.0
        if goal_img is None:
            return np.array([self.alpha * 10.0, 0.0])

        gx, gy = float(goal_img[0]), float(goal_img[1])
        dist   = math.hypot(gx - cx, gy - cy)
        if dist < 1.0:
            return np.array([0.0, 0.0])

        theta = math.atan2(gy - cy, gx - cx)
        F_mag = self.alpha * dist
        return np.array([F_mag * math.cos(theta),
                          F_mag * math.sin(theta)])

    # ── IV-B : Obstacle repulsive force  (Eq. 9) 
    def repulsive_force(self,
                        g_x: np.ndarray,
                        g_y: np.ndarray,
                        ttcs: list,
                        img_h: int,
                        img_w: int) -> np.ndarray:
        """
        F_rep = γ · (1/|R|) · [ ∫_A g(x,y,t) dx  /  Σ_A TTC_i ]   (Eq. 9)

        In discrete form:
            F_rep_x = γ · (1/N) · Σ g_x(x,y)  /  Σ TTC_i
            F_rep_y = γ · (1/N) · Σ g_y(x,y)  /  Σ TTC_i

        A low Σ TTC means things are approaching fast → strong repulsion.
        """
        if not ttcs:
            return np.array([0.0, 0.0])

        sum_ttc = max(sum(ttcs), 1e-4)
        N       = float(img_h * img_w)

        Fx = self.gamma * (float(g_x.sum()) / N) / sum_ttc
        Fy = self.gamma * (float(g_y.sum()) / N) / sum_ttc
        return np.array([Fx, Fy])

    # ── IV-C : Road potential field force  (Eq. 10-18)
    def road_force(self,
                   veh_x    : float,
                   veh_y    : float,
                   foe_img_x: float,
                   img_w    : int) -> np.ndarray:
        """
        Modified Morse potential for lane-keeping  (Eq. 10-18).

        The FOE image position (Sec. IV-C) decides straight vs curved:
          |foe_norm| < 0.15  → straight  (c2 = 0.005, n = 1)
          foe right          → curved right (c2 = +5e-6, n = 2)
          foe left           → curved left  (c2 = −5e-6, n = 2)

        Returns gradient of (Usr + Usl) w.r.t. vehicle_y as (0, Fy).
        """
        dx = 1.0e-10   # small longitudinal offset δx  (Table I)

        foe_norm = (foe_img_x - img_w / 2.0) / (img_w / 2.0)   # [-1, 1]
        if abs(foe_norm) < 0.15:
            c2, n = self.c2_str, 1
        elif foe_norm > 0:
            c2, n = +self.c2_cur, 2
        else:
            c2, n = -self.c2_cur, 2

        c1 = 0.0   # Table I

        xd = veh_x + dx
        yr = c2 * (xd ** n) + c1 * xd + self.c0r   # Eq. 14
        yl = c2 * (xd ** n) + c1 * xd + self.c0l   # Eq. 15

        denom_m = 2.0 * c2 * xd + c1
        m  = -1.0 / denom_m if abs(denom_m) > 1e-8 else -1.0e8
        by = yr - m * xd                             # Eq. 17

        A = self.road_A
        b = self.road_b
        y = veh_y

        def _Usr(yy):
            """Right lane Morse potential  (Eq. 12)"""
            inner_x = (yy - by) / m - xd if abs(m) > 1e-6 else 0.0
            r2 = inner_x ** 2 + (yr - yy) ** 2
            sgn = math.copysign(1.0, yy - yr)
            return A * (1.0 - math.exp(-b * sgn * math.sqrt(max(r2, 1e-12)))) ** 2

        def _Usl(yy):
            """Left lane Morse potential  (Eq. 13)"""
            inner_x = (yy - by) / m - xd if abs(m) > 1e-6 else 0.0
            r2 = inner_x ** 2 + (yl - yy) ** 2
            sgn = math.copysign(1.0, yy - yl)
            return A * (1.0 - math.exp(b * sgn * math.sqrt(max(r2, 1e-12)))) ** 2

        # F_reprd = ∇(Usr + Usl)  (Eq. 18) – numerical derivative w.r.t. y
        dy = 1e-3
        Fy = (_Usr(y + dy) + _Usl(y + dy) -
              _Usr(y - dy) - _Usl(y - dy)) / (2.0 * dy)
        return np.array([0.0, Fy])

    # ── FIX-2 : Hard boundary repulsion
    def boundary_force(self, veh_y: float) -> np.ndarray:
        """
        Strict exponential repulsion from the physical road edges (±1.16 m).

        Activates as soon as the car enters within `bnd_onset` metres of
        either wall and grows exponentially toward bnd_k as the car
        approaches the wall face.  This force is ADDED (not subtracted)
        in compute_desired_heading so it always dominates near the edge.

        Sign convention (world Y-axis, positive = left):
          Near RIGHT wall (veh_y → −1.16): Fy > 0  →  push left
          Near LEFT  wall (veh_y → +1.16): Fy < 0  →  push right
        """
        ROAD_HW = 1.16   # road half-width (m) — matches simulation_setup.py
        Fy = 0.0

        # Distance from each wall; positive means the car is still inside
        dist_right = ROAD_HW + veh_y   # right wall at y = −1.16
        dist_left  = ROAD_HW - veh_y   # left  wall at y = +1.16

        # Right wall → push left (+Fy)
        if dist_right < self.bnd_onset:
            d = max(dist_right, 1e-4)
            Fy += self.bnd_k * math.exp(-d / self.bnd_sigma)

        # Left wall → push right (−Fy)
        if dist_left < self.bnd_onset:
            d = max(dist_left, 1e-4)
            Fy -= self.bnd_k * math.exp(-d / self.bnd_sigma)

        return np.array([0.0, Fy])

    # ── FIX-3 : Lane-centre restoring force (adaptive)
    def centering_force(self, veh_y: float, rep_mag: float = 0.0) -> np.ndarray:
        """
        Proportional attraction toward the lane centre (y = 0), with gain
        that adapts to whether an obstacle is actively being avoided.

        When rep_mag is large (obstacle present):
            effective_k ≈ ctr_k  (weak — don't fight the avoidance)

        When rep_mag is small (open road):
            effective_k ≈ ctr_k * (1 + boost) = up to 5× stronger
            → car is pulled firmly back to y = 0 between obstacles.

        This is the key to recentring: once the car clears an obstacle and
        F_rep drops, centering suddenly dominates and smoothly steers back
        to centre BEFORE the next obstacle is encountered.

        Sign convention: Fy = −effective_k · y
          y > 0 (left of centre)  → Fy < 0  → push right toward centre
          y < 0 (right of centre) → Fy > 0  → push left  toward centre
        """
        # boost decays from 4.0 to 0 as rep_mag rises above ~0.003
        # (rep_mag ≈ 0 on open road; rep_mag grows to ~0.01–0.05 near obstacle)
        boost        = 4.0 * math.exp(-rep_mag / 0.004)
        effective_k  = self.ctr_k * (1.0 + boost)
        return np.array([0.0, -effective_k * veh_y])


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TOTAL FORCE → DESIRED HEADING  (Section V, Eq. 19-21, 27)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_desired_heading(F_att     : np.ndarray,
                             F_rep     : np.ndarray,
                             F_road    : np.ndarray,
                             F_boundary: np.ndarray,
                             F_center  : np.ndarray,
                             psi       : float,
                             lam_X     : float = 0.15,
                             lam_Y     : float = 0.15,
                             lam_bnd   : float = 1.20,
                             lam_ctr   : float = 0.60) -> float:
    """
    Combine image-plane and motion-plane forces into a global desired heading.

    Image-plane total force  (Eq. 19-20, extended with FIX-2 and FIX-3):
        F_XT = F_att_x − F_rep_x − λX · FX_road
        F_YT = F_att_y − F_rep_y − λY · FY_road
                       + λ_bnd · F_boundary_Y      (FIX-2: ADDED, not subtracted)
                       + λ_ctr · F_center_Y         (FIX-3: ADDED, not subtracted)

    The boundary and centering forces are motion-plane forces that are
    ADDED to F_YT (positive sign) because:
      • F_boundary_Y > 0 when near the right wall → should increase F_YT
        (more leftward total force → turn left away from wall).  If it
        were subtracted like F_road it would do the opposite.
      • F_center_Y = −ctr_k·y, so when y < 0 (right of centre)
        F_center_Y > 0 → ADDED → positive F_YT contribution → turns left
        back toward centre.  Correct.

    Both gains (lam_bnd, lam_ctr) are deliberately larger than lam_Y so
    boundary safety and recentring dominate the Morse road term.

    Rotate to global frame  (Eq. 21):
        [F_X']   [ cos ψ   sin ψ ] [F_XT]
        [F_Y'] = [−sin ψ   cos ψ ] [F_YT]

    Desired heading  (Eq. 27):
        ψ_d = atan2(F_Y', F_X')
    """
    F_XT = (  F_att[0]
            - F_rep[0]
            - lam_X * F_road[0]
            + lam_bnd * F_boundary[0]
            + lam_ctr * F_center[0])

    F_YT = (  F_att[1]
            - F_rep[1]
            - lam_Y * F_road[1]
            + lam_bnd * F_boundary[1]
            + lam_ctr * F_center[1])

    cp, sp = math.cos(psi), math.sin(psi)
    F_Xp  =  cp * F_XT + sp * F_YT
    F_Yp  = -sp * F_XT + cp * F_YT

    if abs(F_Xp) < 1e-8 and abs(F_Yp) < 1e-8:
        return psi   # degenerate – hold current heading

    return math.atan2(F_Yp, F_Xp)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  GRADIENT TRACKING SLIDING MODE CONTROLLER  (Section VI, Eq. 28-32)
# ═══════════════════════════════════════════════════════════════════════════════

class GTSMC:
    """
    Lateral controller  (Eq. 28-30):
        ψ_e       = ψ(t) − ψ_d(p)
        s_r       = c_r · ψ_e + ψ̇_e          (rotational sliding manifold)
        u         = −u₀ · tanh(s_r / φ)       ← PROPORTIONAL (was sign)
        δ̇_f       = u   →   δ_f += u · Δt

    Longitudinal controller  (Eq. 31-32):
        s_l       = c_l · v − v_d             (longitudinal sliding manifold)
        a         = −a₀ · sign(s_l)
        v        += a · Δt

    Key changes from bang-bang base:
      • tanh(s_r/φ) replaces sign(s_r)
          - Inside boundary layer |s_r| < φ: steering is PROPORTIONAL to
            heading error — no more full-rate snaps for tiny corrections.
          - Outside boundary layer: saturates to ±u0, same as sign() but
            the transition is smooth, eliminating the chattering that
            caused repeated overshoot.
      • u0        : 0.40 → 0.28   lower peak rate; tanh reaches 0.99·u0
                                   only when s_r ≈ 2.6·φ, so effective
                                   mean rate is much lower than bang-bang.
      • delta_max : 22° → 15°     hard physical limit; 15° at 3.5 m/s
                                   gives a minimum turn radius ≈ 8 m,
                                   safe for a 2.32 m wide road.
      • v_ref     : 5.55 → 3.5    slower speed gives the controller more
                                   time to react before the car travels
                                   its own width laterally.
      • cr        : 3.5 (unchanged) manifold damping stays high.
      • EMA α=0.45 on delta_f: faster than before (was 0.30) because the
          tanh law already suppresses noise; we want quick response.
    """

    # Boundary layer thickness for tanh (rad/s); governs proportional region
    PHI_BL      = 0.30
    # EMA coefficient for delta_f (0 = frozen, 1 = no filter)
    ALPHA_STEER = 0.45

    def __init__(self,
                 cr        = 3.5,
                 u0        = 0.28,                  # peak steering rate (rad/s)
                 cl        = 1.0,
                 a0        = 1.5,
                 v_ref     = 3.5,                   # ~12.6 km/h — more manageable
                 delta_max = math.radians(15)):      # hard wheel-angle limit
        self.cr        = cr
        self.u0        = u0
        self.cl        = cl
        self.a0        = a0
        self.v_ref     = v_ref
        self.delta_max = delta_max

        self._delta_f      = 0.0
        self._delta_f_filt = 0.0
        self._speed        = v_ref
        self._prev_psi_e   = 0.0

    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap angle to (−π, π]."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def update(self, psi: float, psi_d: float,
               speed: float, dt: float):
        """
        Returns (delta_f_smoothed, speed) to send to the vehicle.

        The proportional-tanh control law means:
          • A 1° heading error → s_r ≈ cr·(1°) ≈ 0.061 rad/s
            → u = -u0·tanh(0.061/0.30) ≈ -u0·0.20 = -0.056 rad/s
            → tiny delta_f increment → no snap.
          • A 20° heading error → s_r large → u ≈ -u0 (saturates)
            → maximum steering rate, as needed for real obstacles.
        """
        dt = max(dt, 1e-4)

        # ── Lateral (Eq. 28-30, tanh variant)
        psi_e     = self._wrap(psi - psi_d)
        psi_e_dot = (psi_e - self._prev_psi_e) / dt
        s_r       = self.cr * psi_e + psi_e_dot                # Eq. 28
        u         = -self.u0 * math.tanh(s_r / self.PHI_BL)   # Eq. 30 (tanh)

        # Integrate steering angle
        self._delta_f = float(np.clip(self._delta_f + u * dt,
                                       -self.delta_max, self.delta_max))

        # EMA on delta_f
        self._delta_f_filt = (self.ALPHA_STEER * self._delta_f
                               + (1.0 - self.ALPHA_STEER) * self._delta_f_filt)
        self._prev_psi_e = psi_e

        # ── Longitudinal (Eq. 31-32) 
        s_l         = self.cl * speed - self.v_ref
        a           = -self.a0 * math.copysign(1.0, s_l)
        self._speed = float(np.clip(self._speed + a * dt, 0.5, self.v_ref * 1.2))

        return self._delta_f_filt, self._speed


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DEBUG OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def draw_debug(frame, flows, foe_x, foe_y, goal_img,
               F_att, F_rep, F_road, F_boundary, F_center,
               psi_d, delta_f, speed, veh_y):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Optical flow vectors (green)
    for (x, y, vx, vy) in flows:
        p1 = (int(x), int(y))
        p2 = (int(x + vx * 6), int(y + vy * 6))
        cv2.arrowedLine(frame, p1, p2, (0, 200, 0), 1,
                        cv2.LINE_AA, tipLength=0.4)
        cv2.circle(frame, p1, 2, (0, 200, 0), -1)

    # FOE (cyan cross + rings)
    fx = int(np.clip(foe_x, -w, 2 * w))
    fy = int(np.clip(foe_y, -h, 2 * h))
    for r in (10, 18):
        cv2.circle(frame, (fx, fy), r, (0, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (fx - 18, fy), (fx + 18, fy), (0, 220, 220), 1)
    cv2.line(frame, (fx, fy - 18), (fx, fy + 18), (0, 220, 220), 1)
    cv2.putText(frame, "FOE", (fx + 4, fy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

    # Goal projection (magenta)
    if goal_img is not None:
        cv2.circle(frame, goal_img, 9, (255, 60, 255), -1)
        cv2.putText(frame, "GOAL", (goal_img[0] + 6, goal_img[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 60, 255), 1)

    # Attractive force arrow (blue)
    sc = 180.0
    cv2.arrowedLine(frame, (cx, cy),
                    (int(cx + F_att[0]*sc), int(cy + F_att[1]*sc)),
                    (255, 120, 0), 2, cv2.LINE_AA, tipLength=0.25)

    # Repulsive force arrow (red, reversed so it points away)
    cv2.arrowedLine(frame, (cx, cy),
                    (int(cx - F_rep[0]*sc), int(cy - F_rep[1]*sc)),
                    (40, 40, 255), 2, cv2.LINE_AA, tipLength=0.25)

    # Boundary force arrow (orange) — only drawn when non-trivial
    F_bnd_mag = np.linalg.norm(F_boundary)
    if F_bnd_mag > 0.05:
        cv2.arrowedLine(frame, (cx, cy),
                        (int(cx + F_boundary[0]*sc*0.5),
                         int(cy - F_boundary[1]*sc*0.5)),  # flip Y for screen
                        (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.25)

    # HUD — colour-code boundary warning (red label when active)
    bnd_col = (80, 80, 255) if F_bnd_mag > 0.1 else (255, 255, 255)
    lines = [
        (f"psi_d = {math.degrees(psi_d):+.1f} deg",  (255, 255, 255)),
        (f"delta = {math.degrees(delta_f):+.1f} deg", (255, 255, 255)),
        (f"speed = {speed:.2f} m/s",                  (255, 255, 255)),
        (f"veh_y = {veh_y:+.3f} m",                  (255, 255, 255)),
        (f"flows = {len(flows)}",                      (255, 255, 255)),
        (f"|F_att|  = {np.linalg.norm(F_att):.3f}",  (255, 255, 255)),
        (f"|F_rep|  = {np.linalg.norm(F_rep):.3f}",  (255, 255, 255)),
        (f"|F_road| = {np.linalg.norm(F_road):.3f}", (255, 255, 255)),
        (f"|F_bnd|  = {F_bnd_mag:.3f}",               bnd_col),
        (f"|F_ctr|  = {np.linalg.norm(F_center):.3f}",(255, 255, 255)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (8, 18 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN NAVIGATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class VisualNavigator:
    """
    Orchestrates the full pipeline each frame.
    """

    def __init__(self, gui: bool = True):
        self.env  = SimulationEnvironment(gui=gui)
        self.of   = SparseOpticalFlow(lk_win=25, lk_levels=3,
                                       max_corners=200, quality=0.01,
                                       min_dist=10)
        self.vpf  = VisualPotentialField()
        self.ctrl = GTSMC()
        self._t0  = time.time()

        # FIX-1: EMA state for psi_d smoothing (α = 0.40 → responsive but filtered)
        self._PSI_D_ALPHA   = 0.40
        self._psi_d_filt    = 0.0
        # Hard clamp: psi_d cannot deviate more than this from current yaw
        # Prevents a single bad optical-flow frame from requesting a 90° turn.
        self._MAX_PSI_DELTA = math.radians(15)

    def run(self, max_steps: int = 4000):
        print("Visual Potential Field Navigation — press Q to quit.")

        for step in range(max_steps):

            # ── Step 1 : Capture frame
            frame_bgr, vm, pm = self.env.capture_frame()
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # ── Step 2 : Sparse optical flow  (Sec. III)
            flows = self.of.compute(gray)

            # ── Step 3 : FOE  (Sec. III-A, Eq. 1-3)
            foe_x, foe_y = compute_foe(flows, w, h)

            # ── Step 4 : TTC  (Sec. III-B, Eq. 3)
            ttcs = compute_ttc(flows, foe_x, foe_y)

            # ── Step 5 : Obstacle gradient  (Sec. III-C, Eq. 4-6)
            g_x, g_y = compute_obstacle_gradient(flows, h, w)

            # ── Step 6 : Project goal into image
            goal_img = self.env.project_to_image(self.env.goal_world, vm, pm)

            # ── Step 7 : Visual Potential Field forces  (Sec. IV)
            F_att  = self.vpf.attractive_force(goal_img, w, h)
            F_rep  = self.vpf.repulsive_force(g_x, g_y, ttcs, h, w)
            rep_mag = float(np.linalg.norm(F_rep))

            state  = self.env.get_state()
            veh_y  = float(state['pos'][1])

            F_road     = self.vpf.road_force(
                float(state['pos'][0]), veh_y, foe_x, w
            )
            # FIX-2: hard boundary repulsion (world frame)
            F_boundary = self.vpf.boundary_force(veh_y)
            # FIX-3: adaptive centering — strong when clear, weak near obstacles
            F_center   = self.vpf.centering_force(veh_y, rep_mag)

            # ── Step 8 : Desired heading  (Sec. V, Eq. 19-21, 27)
            psi_d_raw = compute_desired_heading(
                F_att, F_rep, F_road, F_boundary, F_center, state['yaw']
            )

            # Hard-clamp psi_d deviation BEFORE EMA so a single bad
            # optical-flow frame can never command more than MAX_PSI_DELTA
            # of heading change in one step.
            psi_clamp_dev = np.clip(
                GTSMC._wrap(psi_d_raw - state['yaw']),
                -self._MAX_PSI_DELTA, self._MAX_PSI_DELTA
            )
            psi_d_clamped = GTSMC._wrap(state['yaw'] + psi_clamp_dev)

            # EMA on the clamped desired heading (angle-safe wrap)
            dpsi = GTSMC._wrap(psi_d_clamped - self._psi_d_filt)
            self._psi_d_filt = GTSMC._wrap(
                self._psi_d_filt + self._PSI_D_ALPHA * dpsi
            )
            psi_d = self._psi_d_filt

            # ── Step 9 : GTSMC  (Sec. VI, Eq. 28-32) 
            now = time.time()
            dt  = now - self._t0
            self._t0 = now

            delta_f, speed = self.ctrl.update(
                state['yaw'], psi_d, state['speed'], dt
            )

            # ── Step 10 : Apply control to racecar joints  (Eq. 22-26) 
            self.env.apply_bicycle_control(delta_f, speed)
            self.env.step()

            # ── Step 11 : Visualise 
            debug = draw_debug(
                frame_bgr.copy(), flows,
                foe_x, foe_y, goal_img,
                F_att, F_rep, F_road, F_boundary, F_center,
                psi_d, delta_f, speed, veh_y
            )
            cv2.imshow("Visual Potential Field Navigation", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ── Step 12 : Goal check 
            dist = np.linalg.norm(state['pos'][:2] - self.env.goal_world[:2])
            if dist < 1.5:
                print(f"[Step {step}] Goal reached!  Dist = {dist:.2f} m")
                break

            if step % 60 == 0:
                F_ctr_mag = float(np.linalg.norm(F_center))
                print(f"[Step {step:4d}] "
                      f"pos=({state['pos'][0]:.1f},{veh_y:+.2f})  "
                      f"psi={math.degrees(state['yaw']):+.1f}°  "
                      f"psi_d={math.degrees(psi_d):+.1f}°  "
                      f"delta={math.degrees(delta_f):+.1f}°  "
                      f"v={speed:.2f} m/s  "
                      f"goal_dist={dist:.1f} m  "
                      f"|F_rep|={rep_mag:.3f}  "
                      f"|F_bnd|={np.linalg.norm(F_boundary):.2f}  "
                      f"|F_ctr|={F_ctr_mag:.2f}  "
                      f"flows={len(flows)}")

        cv2.destroyAllWindows()
        self.env.disconnect()
        print("Simulation ended.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    nav = VisualNavigator(gui=True)
    nav.run(max_steps=4000)
