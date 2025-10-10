import numpy as np
import cv2

class PostProcessor:
    def __init__(self, tau=7):
        self.tau = tau

    def aimd_filter(self, binary_mask):
        height, width = binary_mask.shape
        filtered_mask = np.zeros_like(binary_mask)
        
        for y in range(height):
            gamma = 0
            for x in range(width):
                s_n = 1 if binary_mask[y, x] > 0 else 0
                if s_n == 1:
                    if gamma < self.tau:
                        gamma += 1
                    else: # gamma == self.tau
                        filtered_mask[y, x] = 255
                else: # s_n == 0
                    gamma = gamma // 2
        
        for y in range(height):
            gamma = 0
            for x in range(width - 1, -1, -1):
                s_n = 1 if binary_mask[y, x] > 0 else 0
                if s_n == 1:
                    if gamma < self.tau:
                        gamma += 1
                    else: # gamma == self.tau
                        filtered_mask[y, x] = 255
                else: # s_n == 0
                    gamma = gamma // 2

        return filtered_mask

    def dynamic_group(self, mask):
        # Ensure the input mask is a uint8 array for cv2.connectedComponentsWithStats
        mask_uint8 = (mask > 0).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels < 2:
            # No foreground components found, return an empty mask of the correct type
            return np.zeros_like(mask, dtype=np.uint8)

        # Return a new mask where all component pixels are set to 255
        return ((labels > 0) * 255).astype(np.uint8)
