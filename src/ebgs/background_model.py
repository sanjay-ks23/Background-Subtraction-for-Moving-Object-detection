import numpy as np

class BackgroundModel:
    def __init__(self, width, height, max_codewords=5, training_frames=20, learning_rate=0.05):
        self.width = width
        self.height = height
        self.max_codewords = max_codewords
        self.training_frames = training_frames
        self.learning_rate = learning_rate

        # Vectorized data structure: (height, width, max_codewords, features)
        # Features: 3 for RGB, 1 for gray, 1 for weight, 1 for last_update
        self.codebooks = np.zeros((height, width, max_codewords, 6), dtype=np.float32)
        self.num_codewords = np.zeros((height, width), dtype=np.int32)

    def process_frame(self, frame, frame_index):
        frame_float = frame.astype(np.float32)
        gray_frame = (frame_float[..., 0] * 0.114 + 
                      frame_float[..., 1] * 0.587 + 
                      frame_float[..., 2] * 0.299)

        if frame_index < self.training_frames:
            self.add_codewords_vectorized(frame_float, gray_frame, frame_index)
            return np.zeros((self.height, self.width), dtype=np.uint8)

        # --- Vectorized Matching ---
        pixel_rgb = frame_float[:, :, np.newaxis, :3]
        pixel_gray = gray_frame[:, :, np.newaxis]
        active_mask = np.arange(self.max_codewords) < self.num_codewords[..., np.newaxis]

        color_dist_sq = np.sum((pixel_rgb - self.codebooks[..., :3])**2, axis=3)
        gray_dist = np.abs(pixel_gray - self.codebooks[..., 3])

        th_sq = 17**2
        tl = 10
        match_mask = (color_dist_sq <= th_sq) & (gray_dist <= tl) & active_mask

        matched_indices = np.argmax(match_mask, axis=2)
        any_match = np.any(match_mask, axis=2)
        
        # --- Vectorized Update ---
        rows, cols = np.where(any_match)
        if rows.size > 0:
            cw_indices = matched_indices[rows, cols]
            self.update_matched_codewords(rows, cols, cw_indices, frame_float, gray_frame, frame_index)

        # --- Handle No-Match Pixels ---
        no_match_rows, no_match_cols = np.where(~any_match)
        if no_match_rows.size > 0:
            self.handle_unmatched_pixels(no_match_rows, no_match_cols, frame_float, gray_frame, frame_index)

        return ~any_match

    def update_matched_codewords(self, rows, cols, cw_indices, frame_float, gray_frame, frame_index):
        # Update RGB values
        self.codebooks[rows, cols, cw_indices, :3] = \
            (1 - self.learning_rate) * self.codebooks[rows, cols, cw_indices, :3] + \
            self.learning_rate * frame_float[rows, cols]
        
        # Update gray values
        self.codebooks[rows, cols, cw_indices, 3] = \
            (1 - self.learning_rate) * self.codebooks[rows, cols, cw_indices, 3] + \
            self.learning_rate * gray_frame[rows, cols]

        # Update weight and last_update
        self.codebooks[rows, cols, cw_indices, 4] += 1
        self.codebooks[rows, cols, cw_indices, 5] = frame_index

    def handle_unmatched_pixels(self, rows, cols, frame_float, gray_frame, frame_index):
        # Pixels where a new codeword can be added
        can_add_mask = self.num_codewords[rows, cols] < self.max_codewords
        add_rows, add_cols = rows[can_add_mask], cols[can_add_mask]
        if add_rows.size > 0:
            add_indices = self.num_codewords[add_rows, add_cols]
            self.set_codeword(add_rows, add_cols, add_indices, frame_float, gray_frame, frame_index)
            self.num_codewords[add_rows, add_cols] += 1

        # Pixels where the oldest codeword must be replaced
        must_replace_mask = ~can_add_mask
        replace_rows, replace_cols = rows[must_replace_mask], cols[must_replace_mask]
        if replace_rows.size > 0:
            last_updates = self.codebooks[replace_rows, replace_cols, :, 5]
            replace_indices = np.argmin(last_updates, axis=1)
            self.set_codeword(replace_rows, replace_cols, replace_indices, frame_float, gray_frame, frame_index)

    def add_codewords_vectorized(self, frame_float, gray_frame, frame_index):
        rows, cols = np.where(self.num_codewords < self.max_codewords)
        if rows.size > 0:
            add_indices = self.num_codewords[rows, cols]
            self.set_codeword(rows, cols, add_indices, frame_float, gray_frame, frame_index)
            self.num_codewords[rows, cols] += 1

    def set_codeword(self, rows, cols, indices, frame_float, gray_frame, frame_index):
        self.codebooks[rows, cols, indices, :3] = frame_float[rows, cols]
        self.codebooks[rows, cols, indices, 3] = gray_frame[rows, cols]
        self.codebooks[rows, cols, indices, 4] = 1.0  # Initial weight
        self.codebooks[rows, cols, indices, 5] = frame_index # last_update