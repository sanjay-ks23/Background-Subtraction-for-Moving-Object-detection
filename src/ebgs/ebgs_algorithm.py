import numpy as np
from .background_model import BackgroundModel
from .post_processing import PostProcessor

class EBGS:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.background_model = BackgroundModel(width, height)
        self.post_processor = PostProcessor()
        self.frame_index = 0

    def process_frame(self, frame):
        initial_mask = self.background_model.process_frame(frame, self.frame_index)

        # Convert the boolean mask to uint8 for post-processing and display
        final_mask = (initial_mask * 255).astype(np.uint8)
        
        if self.frame_index > self.background_model.training_frames:
            # The new background model is efficient, so we apply post-processing after training.
            # The post-processing functions expect a uint8 mask.
            filtered_mask = self.post_processor.aimd_filter(final_mask)
            final_mask = self.post_processor.dynamic_group(filtered_mask)

        self.frame_index += 1
        
        return final_mask
