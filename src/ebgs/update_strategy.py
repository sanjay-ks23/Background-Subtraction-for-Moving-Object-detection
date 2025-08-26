import numpy as np

class AdaptiveUpdater:
    def __init__(self, width, height, block_size=16, min_update_rate=0.05):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.min_update_rate = min_update_rate
        
        self.num_blocks_x = (width + self.block_size - 1) // self.block_size
        self.num_blocks_y = (height + self.block_size - 1) // self.block_size
        
        self.update_rates = np.full((self.num_blocks_y, self.num_blocks_x), min_update_rate)
        self.circular_queue = np.random.randint(0, self.block_size * self.block_size, size=self.block_size * self.block_size)
        self.queue_start_index = 0

    def calculate_update_rates(self, frame, background_model):
        # This is a simplified version of the paper's approach
        # A full implementation would require calculating false alarm rates per block
        pass

    def update_background_model(self, frame, background_model, frame_index):
        for y_block in range(self.num_blocks_y):
            for x_block in range(self.num_blocks_x):
                alpha = self.update_rates[y_block, x_block]
                num_pixels_to_update = int(alpha * self.block_size * self.block_size)
                
                start_y = y_block * self.block_size
                start_x = x_block * self.block_size
                
                end_y = min(start_y + self.block_size, self.height)
                end_x = min(start_x + self.block_size, self.width)

                update_indices = self.circular_queue[self.queue_start_index : self.queue_start_index + num_pixels_to_update]
                self.queue_start_index = (self.queue_start_index + num_pixels_to_update) % len(self.circular_queue)

                for i in update_indices:
                    y_offset = i // self.block_size
                    x_offset = i % self.block_size
                    
                    y = start_y + y_offset
                    x = start_x + x_offset

                    if y < end_y and x < end_x:
                        pixel_data = frame[y, x]
                        codebook = background_model.codebooks[y, x]
                        codebook.match_and_update(pixel_data, frame_index, background_model.learning_rate, 10, 17)
