class Ball:
    def __init__(self, x0,y0,x1,y1):
        self.bbx = (x0,y0,x1,y1)
        self.real_position = None

    def get_bbx(self):
        return self.bbx
    
    def update_position(self, bbx):
        self.bbx = bbx
    