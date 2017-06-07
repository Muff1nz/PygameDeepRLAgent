class Settings():
    def __init__(self):
        self.screenWidth = 1024
        self.screenHeight = 1024
        self.gameSecond = 60 # Amount of frames considered a second in game
        self.fps = 60 # Maximum fps for the game
        self.mspf = 1 / self.fps * 1000 # miliseconds per frame