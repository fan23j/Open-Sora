class MicroBatch:
    
    def __init__(self, index: int, num_frames: int, height: int, width: int):
        self.index = index
        self.num_frames = num_frames
        self.height = height
        self.width = width