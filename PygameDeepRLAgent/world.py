import pygame

BLACK = 0, 0, 0


class World:
    def __init__(self):
        self.walls = []
        self.walls.append(Line([0, 500], [500, 500]))


    def draw(self, screen):
        for wall in self.walls:
            wall.draw(screen)



class Line:
    def __init__(self, start, end):
        self.clr = BLACK
        self.start = start
        self.end = end
        a = (end[1] - start[1]) / (end[0] - start[0])
        self.line = lambda x, y: a*(x - self.start[0]) - (y - self.start[1])

    def draw(self, screen):
        pygame.draw.line(screen, self.clr, self.start, self.end, 30)




