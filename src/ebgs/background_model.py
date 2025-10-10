import numpy as np

class BackgroundModel:
    def __init__(self, width, height, training_frames=20, learning_rate=0.05):
        self.width = width
        self.height = height
        self.training_frames = training_frames
        self.learning_rate = learning_rate
        self.codebooks = np.empty((height, width), dtype=object)
        for y in range(height):
            for x in range(width):
                self.codebooks[y, x] = self.Codebook()

    class Codebook:
        def __init__(self, max_codewords=5):
            self.codewords = []
            self.max_codewords = max_codewords

        def match_and_update(self, pixel_data, frame_index, learning_rate, tl, th):
            r, g, b = pixel_data
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b

            matched_codeword = None
            for codeword in self.codewords:
                color_dist = np.sqrt(
                    (r - codeword['rgb'][0])**2 +
                    (g - codeword['rgb'][1])**2 +
                    (b - codeword['rgb'][2])**2
                )
                if color_dist <= th and abs(gray_value - codeword['gray']) <= tl:
                    matched_codeword = codeword
                    break

            if matched_codeword:
                matched_codeword['rgb'] = (1 - learning_rate) * matched_codeword['rgb'] + learning_rate * np.array([r, g, b])
                matched_codeword['gray'] = (1 - learning_rate) * matched_codeword['gray'] + learning_rate * gray_value
                matched_codeword['weight'] += 1
                matched_codeword['last_update'] = frame_index
                return 0

            if len(self.codewords) < self.max_codewords:
                self.add_codeword(pixel_data, frame_index)
            else:
                last_updates = [cw['last_update'] for cw in self.codewords]
                oldest_index = np.argmin(last_updates)
                self.codewords.pop(oldest_index)
                self.add_codeword(pixel_data, frame_index)
            return 1

        def add_codeword(self, pixel_data, frame_index):
            r, g, b = pixel_data
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            new_codeword = {
                'rgb': np.array([r, g, b]),
                'gray': gray_value,
                'weight': 1,
                'last_update': frame_index
            }
            self.codewords.append(new_codeword)

    def process_frame(self, frame, frame_index):
        foreground_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        tl = 10
        th = 17

        for y in range(self.height):
            for x in range(self.width):
                pixel_data = frame[y, x]
                codebook = self.codebooks[y, x]
                
                if frame_index < self.training_frames:
                    codebook.add_codeword(pixel_data, frame_index)
                else:
                    is_foreground = codebook.match_and_update(pixel_data, frame_index, self.learning_rate, tl, th)
                    if is_foreground:
                        foreground_mask[y, x] = 255
        
        return foreground_mask
