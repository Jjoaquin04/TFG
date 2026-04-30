class Ball:
    def __init__(self):
        self.bbx = None
        self.real_position = None

    def get_bbx(self):
        return self.bbx

    def get_real_position(self):
        return self.real_position
    
    def update(self, bbx, real_position):
        self.bbx = bbx
        self.real_position = real_position

    