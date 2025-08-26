import numpy as np
from .background_model import BackgroundModel
from .post_processing import PostProcessor
from .update_strategy import AdaptiveUpdater

class EBGS:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.background_model = BackgroundModel(width, height)
        self.post_processor = PostProcessor()
        self.updater = AdaptiveUpdater(width, height)
        self.frame_index = 0

    def process_frame(self, frame):
        initial_mask = self.background_model.process_frame(frame, self.frame_index)
        
        if self.frame_index > self.background_model.training_frames:
            filtered_mask = self.post_processor.aimd_filter(initial_mask)
            final_mask = self.post_processor.dynamic_group(filtered_mask)
            
            self.updater.calculate_update_rates(frame, self.background_model)
            self.updater.update_background_model(frame, self.background_model, self.frame_index)
        else:
            final_mask = initial_mask

        self.frame_index += 1
        
        # Convert boolean mask to uint8 for display
        final_mask_display = (final_mask * 255).astype(np.uint8)
        
        return final_mask_display
