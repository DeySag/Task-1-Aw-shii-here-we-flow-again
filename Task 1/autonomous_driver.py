import pybullet as p
import cv2
import numpy as np
import time

from simulation_setup import setup_simulation
from lucaskanade import LucasKanadeTracker

def get_ego_camera_frame(car_id, width=400, height=400):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    yaw = p.getEulerFromQuaternion(orn)[2]
    
    cam_x = pos[0] + 0.2 * np.cos(yaw)
    cam_y = pos[1] + 0.2 * np.sin(yaw)
    cam_pos = [cam_x, cam_y, pos[2] + 0.3]
    
    target_pos = [cam_x + 5.0 * np.cos(yaw), cam_y + 5.0 * np.sin(yaw), pos[2] + 0.3]
    
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(60.0, float(width)/height, 0.1, 100.0)
    
    images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)
    rgba_img = np.reshape(images[2], (height, width, 4)).astype(np.uint8)
    return cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)

def main():
    print("Setting up world...")
    car_id, steering_joints, motor_joints = setup_simulation(gui=True)
    
    lk_tracker = LucasKanadeTracker()
    
    # FIX 1: Increase box size to catch faster, closer pixels
    box_size = 25 
    image_width, image_height = 400, 400
    
    x0_arr, y0_arr = [], []
    prev_gray = None
    frame_counter = 0

    # SMOOTHING & CONTROL VARIABLES
    smooth_foe_x = image_width / 2
    smooth_foe_y = image_height / 2
    
    # FIX 2: Add steering memory so it doesn't snap back when blind
    smooth_steer = 0.0 

    K_att = 0.008     

    dt = 1.0 / 60.0
    try:
        while True:
            for j in motor_joints:
                p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=12.0, force=800)
            
            p.stepSimulation()
            frame = get_ego_camera_frame(car_id, image_width, image_height)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # FIX 3: Refresh features twice as fast (every 4 frames)
            if prev_gray is None or frame_counter % 4 == 0:
                corners = cv2.goodFeaturesToTrack(curr_gray, maxCorners=60, qualityLevel=0.05, minDistance=15)
                if corners is not None:
                    x0_arr, y0_arr = [], [] 
                    for corner in corners:
                        x, y = corner.ravel()
                        tx, ty = int(x - box_size/2), int(y - box_size/2)
                        if 0 <= tx < image_width - box_size and 0 <= ty < image_height - box_size:
                            x0_arr.append(tx)
                            y0_arr.append(ty)
            
            valid_flow_vectors = []
            
            if prev_gray is not None and len(x0_arr) > 0:
                reduced_prev = prev_gray.astype(float)
                reduced_curr = curr_gray.astype(float)
                
                A_matrix, B_matrix = [], []
                
                for i in range(len(x0_arr) - 1, -1, -1):
                    cx, cy = x0_arr[i], y0_arr[i]
                    
                    if cx <= 0 or cy <= 0 or cx >= image_width - box_size or cy >= image_height - box_size:
                        x0_arr.pop(i); y0_arr.pop(i)
                        continue

                    changes_x = lk_tracker.Ix(reduced_prev, cy, cx, box_size)
                    changes_y = lk_tracker.Iy(reduced_prev, cy, cx, box_size)
                    changes_t = lk_tracker.It(reduced_prev, reduced_curr, cy, cx, box_size)

                    s_matrix = np.column_stack((changes_x.flatten(), changes_y.flatten()))
                    st_s = np.matmul(s_matrix.T, s_matrix)

                    try:
                        st_s_inv = np.linalg.inv(st_s)
                        vel = np.matmul(np.matmul(st_s_inv, s_matrix.T), changes_t.flatten())
                        u, v = vel[0], vel[1]
                        
                        x0_arr[i] += int(round(u))
                        y0_arr[i] += int(round(v))
                        
                        center_x, center_y = int(cx + box_size/2), int(cy + box_size/2)
                        cv2.line(frame, (center_x, center_y), (int(center_x + u*5), int(center_y + v*5)), (0, 0, 255), 2)
                        
                        valid_flow_vectors.append((center_x, center_y, u, v))
                        
                        if abs(u) > 0.5 or abs(v) > 0.5:
                            A_matrix.append([v, -u])
                            B_matrix.append(v * center_x - u * center_y)

                    except np.linalg.LinAlgError:
                        x0_arr.pop(i); y0_arr.pop(i)

           # ==============================================================
            # 4. PAPER-ACCURATE VISUAL POTENTIAL FIELD (Capito et al. 2020)
            # ==============================================================
            attractive_force = 0.0
            repulsive_force = 0.0
            guardrail_force = 0.0  

            # --- A. ATTRACTIVE FIELD (The Target) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            target_mask = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
            target_mask[int(image_height * 0.6):, :] = 0 
            M = cv2.moments(target_mask)
            
            target_x = int(image_width / 2.0) 
            if M["m00"] > 0:
                target_x = int(M["m10"] / M["m00"])
                target_y = int(M["m01"] / M["m00"])
                cv2.circle(frame, (target_x, target_y), 15, (255, 0, 0), -1)
                
                K_att = 0.008 
                attractive_force = ((image_width / 2.0) - target_x) * K_att 
            else:
                attractive_force = -smooth_steer * 0.5 

            # --- B. REPULSIVE FIELD (TTC Based - Eq 3) ---
            danger_zone_left = 120
            danger_zone_right = 280
            danger_zone_top = 160
            danger_zone_bottom = 260 
            cv2.rectangle(frame, (danger_zone_left, danger_zone_top), 
                         (danger_zone_right, danger_zone_bottom), (0, 0, 100), 2)

            K_rep_ttc = 5.0 # Gain for Time-To-Contact repulsion

            for (cx, cy, u, v) in valid_flow_vectors:
                mag = np.sqrt(u**2 + v**2)
                
                if mag > 0.5 and danger_zone_top < cy < danger_zone_bottom:
                    if danger_zone_left < cx < danger_zone_right:
                        
                        # Calculate Time To Contact (TTC) relative to FOE
                        dist_to_foe = np.sqrt((cx - smooth_foe_x)**2 + (cy - smooth_foe_y)**2)
                        ttc = dist_to_foe / (mag + 1e-5) # Prevent division by zero
                        
                        # If collision is imminent (TTC is low), push hard!
                        if ttc < 50.0: 
                            # Dodge relative to target to enable slaloming
                            dist_from_target = cx - target_x 
                            
                            # Force is inversely proportional to TTC
                            rep_push = np.sign(dist_from_target) * (1.0 / ttc) * K_rep_ttc
                            repulsive_force += rep_push
                            
                            end_push_x = int(cx + rep_push * 50)
                            cv2.line(frame, (cx, cy), (end_push_x, cy), (0, 255, 255), 3)

            # --- C. ROAD POTENTIAL FIELD (Morse Potential - Eq 11) ---
            # Parameters from the paper (Table I) adjusted for pixel scale
            A_morse = 0.5  
            b_morse = 0.05 
            guardrail_margin = 80
            
            cv2.line(frame, (guardrail_margin, int(image_height/2)), (guardrail_margin, image_height), (255, 0, 0), 2)
            cv2.line(frame, (image_width - guardrail_margin, int(image_height/2)), (image_width - guardrail_margin, image_height), (255, 0, 0), 2)

            for (cx, cy, u, v) in valid_flow_vectors:
                mag = np.sqrt(u**2 + v**2)
                
                if mag > 1.0 and cy > image_height / 2:
                    # Left Lane Boundary
                    if cx < guardrail_margin:
                        dist_to_edge = guardrail_margin - cx
                        # Derivative of Morse Potential: 2 * A * b * e^(-b*d) * (1 - e^(-b*d))
                        morse_force = 2 * A_morse * b_morse * np.exp(-b_morse * dist_to_edge) * (1 - np.exp(-b_morse * dist_to_edge))
                        
                        # Steer right (-)
                        push = -morse_force * mag * 10.0
                        guardrail_force += push
                        cv2.line(frame, (cx, cy), (int(cx + push * 100), cy), (255, 0, 255), 4)
                        
                    # Right Lane Boundary
                    elif cx > image_width - guardrail_margin:
                        dist_to_edge = cx - (image_width - guardrail_margin)
                        morse_force = 2 * A_morse * b_morse * np.exp(-b_morse * dist_to_edge) * (1 - np.exp(-b_morse * dist_to_edge))
                        
                        # Steer left (+)
                        push = morse_force * mag * 10.0
                        guardrail_force += push
                        cv2.line(frame, (cx, cy), (int(cx + push * 100), cy), (255, 0, 255), 4)

            # --- APPLY FINAL STEERING ---
            raw_steering_angle = attractive_force + repulsive_force + guardrail_force
            smooth_steer = (0.7 * raw_steering_angle) + (0.3 * smooth_steer)
            safe_steer = max(-0.4, min(0.4, smooth_steer))
            
            for j in steering_joints:
                p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=safe_steer)

            # UI Telemetry
            cv2.putText(frame, f"Att: {attractive_force:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Rep (TTC): {repulsive_force:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Grd (Morse): {guardrail_force:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Steer: {safe_steer:.3f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # ==============================================================

            cv2.imshow("Optical Flow Agent Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            prev_gray = curr_gray.copy()
            frame_counter += 1
            time.sleep(dt)
            
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()