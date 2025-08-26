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
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels < 2:
            return mask

        return labels > 0
