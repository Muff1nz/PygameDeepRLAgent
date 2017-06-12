import pygame

BLACK = 0, 0, 0

class World:
    def __init__(self, settings):
        w = settings.screenWidth
        h = settings.screenHeight

        self.walls = []

        #4 walls
        self.walls.append(Line([0, h/5], [0, h  - h / 5]))
        self.walls.append(Line([w, h / 5], [w, h - h / 5]))
        self.walls.append(Line([w / 5, 0], [w - w / 5, 0]))
        self.walls.append(Line([w / 5, h], [w - w / 5, h]))

        #4 corners
        self.walls.append(Line([w - w / 5, 0], [w, h / 5]))
        self.walls.append(Line([0, h / 5], [w / 5, 0]))
        self.walls.append(Line([0, h - h / 5], [w / 5, h]))
        self.walls.append(Line([w - w / 5, h], [w, h - h / 5]))


    def draw(self, screen):
        for wall in self.walls:
            wall.draw(screen)



class Line:
    def __init__(self, start, end):
        self.clr = BLACK
        self.start = start
        self.end = end
        if end[0] != start[0] and end[1] != start[1]: # normal lines
            self.straight = False
            a = (end[1] - start[1]) / (end[0] - start[0])
            self.line = lambda x, y: a*(x - self.start[0]) - (y - self.start[1])
        else: # Vertical/horizontal lines
            self.straight = True
            if end[0] == start[0]:
                self.line = lambda x, y: end[0] - x
            else:
                self.line = lambda x, y: end[1] - y

    def draw(self, screen):
        pygame.draw.line(screen, self.clr, self.start, self.end, 30)




