class MiniCourt():

    def __init__(self, frame_height):

        self.rectangle_width = 250
        self.rectangle_height = 450
        self.margin = 50
        self.padding = 30
    
        self.set_background_court(frame_height)
    
    def set_mini_court(self, frame_height):

        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y - self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y + self.padding
        

    def set_background_court(self, frame_height):

        self.start_x = self.margin 
        self.start_y = frame_height
        self.end_x = self.start_x + self.rectangle_width
        self.end_y = self.start_y - self.rectangle_height

