import numpy as np
import pygame


class physicsHandler():
    def __init__(self, world, player, targetHandler, settings):
        self.settings = settings
        self.world = world
        self.player = player
        self.target = targetHandler.target
        self.events = []
        self.quadTree = None

    # Checks for collisions between objects in game and resolves them
    def update(self, playerTimeStep):
        # Make quadTree
        bullets = []

        for bullet in self.player.ws.bullets:
            if bullet.active:
                bullets.append(bullet)

        GE = GameEntities(self.player, bullets, self.target, self.world.walls)
        self.quadTree = QuadTree(GE, self.settings)
        self.traverseAndCheckQuads(self.quadTree, playerTimeStep)

    # Recursively traverses quad tree and performs collision checking on each leaf node
    def traverseAndCheckQuads(self, quad, playerTimeStep):
        if len(quad.nodes):
            for q in quad.nodes:
                self.traverseAndCheckQuads(q, playerTimeStep)
        else:
            self.checkCollisions(quad.GE, playerTimeStep)

    # Collision checking performed on one node in quadtree
    def checkCollisions(self, GE, playerTimeStep):
        # collisions for player
        if GE.player:
            boxWallCollision(GE.player, GE.walls)
            for bullet in GE.bullets:
                if GE.target:
                    if boxCollision(bullet, GE.target):
                        GE.target.playerBulletCollision()
                        bullet.wallCollision()
                        if self.settings.causalityTracking:
                            self.events.append(["Player hit target!", bullet.playerTimeStep])
                        else:
                            self.events.append(["Player hit target!", playerTimeStep])
                if boxWallCollision(bullet, GE.walls):
                    bullet.wallCollision()


# ============================Helper functions========================================
def boxWallCollision(box, walls):
    for wall in walls:
        if boxLineCollision(wall, box):
            box.pos = box.oldPos.copy()
            return True
    return False


# returns true if two boxes are colliding
def boxCollision(box1, box2):
    if (box1.pos[1] + box1.size <= box2.pos[1] or
                box1.pos[1] >= box2.pos[1] + box2.size or
                    box1.pos[0] + box1.size <= box2.pos[0] or
                box1.pos[0] >= box2.pos[0] + box2.size):
        return False
    return True


# returns true if a box and a line is colliding
def boxLineCollision(line, box):
    sign = 0
    boundsCount = 0
    for vertex in box.vertices:
        sign += np.sign(line.line(vertex[0] + box.pos[0], vertex[1] + box.pos[1]))
        if checkLineBounds(line, vertex + box.pos) or line.straight:
            boundsCount += 1
    if abs(sign) != 4 and boundsCount != 0:
        return True
    return False


# returns true if a point is inside the bounds of a line
def checkLineBounds(line, point):
    offset = 1
    y = [line.start[1], line.end[1]]
    x = [line.start[0], line.end[0]]
    bounds = []
    xBound = (max(x) + offset) > point[0] > (min(x) - offset)
    yBound = (max(y) + offset) > point[1] > (min(y) - offset)
    if xBound and yBound:  # Point is inside the range of the line
        return True
    else:
        return False  # Point is outside range of line


# ==========================Helper classes============================
class Box:
    def __init__(self, pos=np.zeros(shape=[2]), size=-1):
        self.pos = pos
        self.size = size
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.size]))
        self.vertices.append(np.array([self.size, 0]))
        self.vertices.append(np.array([self.size, self.size]))


class GameEntities():
    def __init__(self, player, bullets, target, walls):
        self.player = player
        self.bullets = bullets
        self.target = target
        self.walls = walls
        self.count = len(self.bullets) + len(walls)
        if player:
            self.count += 1
        if target:
            self.count += 1


class QuadTree():
    def __init__(self, GE, settings, box=Box(), depth=0):
        self.settings = settings
        self.GE = GE
        self.depth = depth
        self.box = box
        if box.size == -1:
            box.size = settings.screenRes
        self.rect = pygame.Rect(self.box.pos[0], self.box.pos[1], self.box.size, self.box.size)
        self.nodes = []
        if GE.count > settings.quadTreeMaxObjects and self.depth < settings.quadTreeDepth:
            self.split()

    # Splits one quad into 4 smaller quads
    def split(self):
        positions = np.array([
            [0, 0],
            [0, self.box.size / 2],
            [self.box.size / 2, 0],
            [self.box.size / 2, self.box.size / 2]
        ])
        positions = np.add(positions[..., :2], self.box.pos)
        size = self.box.size / 2
        for i in range(4):
            box = Box(positions[i], size)
            self.nodes.append(QuadTree(self.assignObjectsToQuad(box), self.settings, box, self.depth + 1))

    def assignObjectsToQuad(self, box):
        player = None
        target = None
        bullets = []
        walls = []

        if self.GE.player:
            if boxCollision(self.GE.player, box):
                player = self.GE.player

        if self.GE.target:
            if boxCollision(self.GE.target, box):
                target = self.GE.target

        for bullet in self.GE.bullets:
            if boxCollision(box, bullet):
                bullets.append(bullet)

        for wall in self.GE.walls:
            if boxLineCollision(wall, box):
                walls.append(wall)

        GE = GameEntities(player, bullets, target, walls)
        return GE

    def draw(self, screen):
        if not len(self.nodes):
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 1)
        else:
            for node in self.nodes:
                node.draw(screen)




