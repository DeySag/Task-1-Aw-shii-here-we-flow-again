import numpy as np
import cv2

class LucasKanadeTracker:
    def __init__(self, num_pyramid_levels=3, num_iterations=5):
        """
        Initialize the Lucas-Kanade tracker with pyramid parameters.
        
        Args:
            num_pyramid_levels: Number of pyramid levels (default: 3)
            num_iterations: Number of iterations per pyramid level (default: 5)
        """
        self.num_pyramid_levels = num_pyramid_levels
        self.num_iterations = num_iterations

    def build_pyramid(self, img, num_levels):
        """
        Build a Gaussian pyramid from an image.
        
        Args:
            img: Input image
            num_levels: Number of pyramid levels
            
        Returns:
            List of images at different scales
        """
        pyramid = [img]
        for i in range(1, num_levels):
            downsampled = cv2.pyrDown(pyramid[i-1])
            pyramid.append(downsampled)
        return pyramid

    def Ix(self, grid, row, col, box_size):
        # Grab the window
        window = grid[row : row + box_size, col : col + box_size]
    
        # Prepare an empty array for the result
        ix = np.zeros_like(window, dtype=float)
    
        # [Internal pixels] = (Pixels to the Right - Pixels to the Left) / 2
        right_neighbors = window[:, 2:]  # All rows, starting from index 2 to end
        left_neighbors  = window[:, :-2] # All rows, from start to 2 pixels before end
    
        ix[:, 1:-1] = (right_neighbors - left_neighbors) / 2.0
    
        # 4. Handle edges
        ix[:, 0]  = (window[:, 1] - window[:, 0]) / 2.0  # Left edge
        ix[:, -1] = (window[:, -1] - window[:, -2]) / 2.0 # Right edge
    
        return ix

    def Iy(self, grid, row, col, box_size):
        # Grab the window
        window = grid[row : row + box_size, col : col + box_size]
        iy = np.zeros_like(window, dtype=float)
    
        # [Internal pixels] = (Row Below - Row Above) / 2.0
        below_neighbors = window[2:, :]  # From row 2 to the end
        above_neighbors = window[:-2, :] # From start to 2 rows before the end
    
        iy[1:-1, :] = (below_neighbors - above_neighbors) / 2.0

        # Handle top and bottom boundaries
        iy[0, :]  = (window[1, :] - window[0, :]) / 2.0  # Top edge
        iy[-1, :] = (window[-1, :] - window[-2, :]) / 2.0 # Bottom edge

        return iy

    def It(self, frame1, frame2, row, col, box_size):
        # Extract the same box area from both frames
        window1 = frame1[row : row + box_size, col : col + box_size]
        window2 = frame2[row : row + box_size, col : col + box_size]

        # The temporal gradient: flip to (frame1 - frame2) to get motion direction correct
        it = window1.astype(float) - window2.astype(float)

        return it

    def compute_flow_at_level(self, frame1, frame2, row, col, box_size, 
                             initial_velocity=None, scale_factor=1.0):
        """
        Compute optical flow for a single iteration at one pyramid level.
        
        Args:
            frame1: Previous frame at this pyramid level
            frame2: Current frame at this pyramid level
            row, col: Position of window in this pyramid level
            box_size: Size of the window
            initial_velocity: Initial velocity estimate (for coarse-to-fine refinement)
            scale_factor: Scale factor relative to original image
            
        Returns:
            Velocity vector as 2x1 numpy array
        """
        changes_x = self.Ix(frame1, row, col, box_size)
        changes_y = self.Iy(frame1, row, col, box_size)
        changes_t = self.It(frame1, frame2, row, col, box_size)

        s_matrix = np.concatenate(np.array([changes_x.flatten()[None].T, 
                                           changes_y.flatten()[None].T]), axis=1)
        s_matrix_transpose = s_matrix.transpose()
        st_s = np.matmul(s_matrix_transpose, s_matrix)

        try:
            st_s_inv = np.linalg.inv(st_s)
            temp_matrix = np.matmul(st_s_inv, s_matrix_transpose)
            velocity = -np.matmul(temp_matrix, changes_t.flatten()[None].T)
        except np.linalg.LinAlgError:
            velocity = np.array([[0.0], [0.0]])

        return velocity

    def pyramidal_lucas_kanade(self, frame1, frame2, row, col, box_size):
        """
        Compute optical flow using pyramidal Lucas-Kanade algorithm.
        
        Args:
            frame1: Previous frame
            frame2: Current frame
            row, col: Position of window in original image
            box_size: Size of the window
            
        Returns:
            Velocity vector (vx, vy) as 2x1 numpy array
        """
        # Build pyramids for both frames
        pyramid1 = self.build_pyramid(frame1, self.num_pyramid_levels)
        pyramid2 = self.build_pyramid(frame2, self.num_pyramid_levels)
        
        # Initialize velocity as zero at the coarsest level
        velocity = np.array([[0.0], [0.0]])
        
        # Start from the coarsest level and work down to the finest
        for level in range(self.num_pyramid_levels - 1, -1, -1):
            # Scale position to current level
            scale_factor = 2 ** level
            scaled_row = row // scale_factor
            scaled_col = col // scale_factor
            
            # Get frames at current pyramid level
            frame1_level = pyramid1[level]
            frame2_level = pyramid2[level]
            
            # Ensure position is valid
            scaled_row = max(0, min(scaled_row, frame1_level.shape[0] - box_size - 1))
            scaled_col = max(0, min(scaled_col, frame1_level.shape[1] - box_size - 1))
            
            # Compute optical flow at this level (single computation, no iteration/warping)
            delta_velocity = self.compute_flow_at_level(
                frame1_level, frame2_level,
                scaled_row, scaled_col, box_size
            )
            
            # Accumulate the incremental velocity
            velocity = velocity + delta_velocity
            
            # Scale velocity up for next finer level
            if level > 0:
                velocity = velocity * 2.0
        
        return velocity