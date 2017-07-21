import pygame

BLACK = 0, 0, 0

class World:
    def __init__(self, settings):
        res = settings.screenRes

        self.walls = []

        #4 walls
        self.walls.append(Line([0, res/5], [0, res - res / 5]))
        self.walls.append(Line([res, res / 5], [res, res - res / 5]))
        self.walls.append(Line([res / 5, 0], [res - res / 5, 0]))
        self.walls.append(Line([res / 5, res], [res - res / 5, res]))

        #4 corners
        self.walls.append(Line([res - res / 5, 0], [res, res / 5]))
        self.walls.append(Line([0, res / 5], [res / 5, 0]))
        self.walls.append(Line([0, res - res / 5], [res / 5, res]))
        self.walls.append(Line([res - res / 5, res], [res, res - res / 5]))

        # Center piece
        cs = 5 #center size
        self.walls.append(Line([res / 2 - res / cs, res / 2], [res / 2, res / 2 + res / cs]))
        self.walls.append(Line([res / 2, res / 2 + res / cs], [res / 2 + res / cs, res / 2]))
        self.walls.append(Line([res / 2, res / 2 - res / cs], [res / 2 + res / cs, res / 2]))
        self.walls.append(Line([res / 2 - res / cs, res / 2], [res / 2, res / 2 - res / cs]))

        #Navigation nodes:
        self.nodes = []
        self.nodes.append(Node([res * 0.2, res * 0.2]))
        self.nodes.append(Node([res * 0.8, res * 0.2]))
        self.nodes.append(Node([res * 0.8, res * 0.8]))
        self.nodes.append(Node([res * 0.2, res * 0.8]))
        for i, node in enumerate(self.nodes):
            node.neighbors.append(self.nodes[(i + 1) % len(self.nodes)])
            node.neighbors.append(self.nodes[(i - 1) % len(self.nodes)])

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
        pygame.draw.line(screen, self.clr, self.start, self.end, 50)

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.neighbors = []