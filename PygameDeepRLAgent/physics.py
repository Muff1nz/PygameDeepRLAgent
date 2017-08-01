import numpy as np
import pygame

class physicsHandler():
    def __init__(self, world, player, enemies, settings):
        self.settings = settings
        self.world = world
        # Array of boxes, could be a player, bullets, enemies...
        self.player = player
        self.enemies = enemies.enemies
        self.events = []
        self.collisionChecks = 0
        self.quadTree = None

    # Checks for collisions between objects in game and resolves them
    def update(self, ei):
        if self.quadTree:
            self.quadTree.clear()
        # Make quadTree
        playerBullets = []
        enemies = []
        enemyBullets = []

        for bullet in self.player.ws.bullets:
            if bullet.active:
                playerBullets.append(bullet)

        for enemy in self.enemies:
            if enemy.active:
                enemies.append(enemy)
            for bullet in enemy.ws.bullets:
                if bullet.active:
                    enemyBullets.append(bullet)

        GE = GameEntities(self.player, playerBullets, enemies, enemyBullets, self.world.walls)
        self.quadTree = QuadTree(GE, self.settings)

        self.collisionChecks = 0
        self.traverseAndCheckQuads(self.quadTree, ei)

    def traverseAndCheckQuads(self, quad, ei):
        if len(quad.nodes):
            for q in quad.nodes:
                self.traverseAndCheckQuads(q, ei)
        else:
            self.checkCollisions(quad.GE, ei)

    def checkCollisions(self, GE, ei):
        # collisions for player
        if GE.player:
            boxWallCollision(GE.player, GE.walls)
            self.collisionChecks += 1

        for bullet in GE.playerBullets:
            boxWallCollision(bullet, GE.walls)
            self.collisionChecks += 1
            for enemy in GE.enemies:
                if boxCollision(bullet, enemy):
                    enemy.kill()
                    self.events.append(["Enemy killed", bullet.ei])
                self.collisionChecks += 1

        # collisions for enemies
        for enemy in GE.enemies:
            boxWallCollision(enemy, GE.walls)
            self.collisionChecks += 1

        for bullet in GE.enemyBullets:
            boxWallCollision(bullet, GE.walls)
            self.collisionChecks += 1
            if GE.player:
                if boxCollision(bullet, GE.player):
                    self.events.append(["Player killed", ei])
                self.collisionChecks += 1

#============================Helper functions========================================
def boxWallCollision(box, walls):
    for wall in walls:
        if boxLineCollision(wall, box):
            box.pos = box.oldPos.copy()
            if box.type == "bullet":  # Make bullets bounce off of walls
                if wall.straight:  # Straight wall, reverse direction
                    box.dir *= -1
                else:
                    box.dir = box.dir[::-1]  # Diagonal wall, swap x and y direction
                    if wall.start[1] > wall.end[1]:  # Descending wall, reverse after swap
                        box.dir *= -1

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

#==========================Helper classes============================
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
    def __init__(self, player, playerBullets, enemies, enemyBullets, walls):
        self.player = player
        self.playerBullets = playerBullets
        self.enemies = enemies
        self.enemyBullets = enemyBullets
        self.walls = walls
        self.count = len(playerBullets) + len(enemyBullets) + len(enemies) + len(walls)
        if player:
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

    def split(self):
        positions = np.array([
            [0, 0],
            [0, self.box.size/2],
            [self.box.size/2, 0],
            [self.box.size/2, self.box.size/2]
        ])
        positions = np.add(positions[..., :2], self.box.pos)
        size = self.box.size/2
        for i in range(4):
            box = Box(positions[i], size)
            self.nodes.append(QuadTree(self.assignObjectsToQuad(box), self.settings, box, self.depth + 1))

    def assignObjectsToQuad(self, box):
        player = None
        playerBullets = []
        enemies = []
        enemyBullets = []
        walls = []

        if self.GE.player:
            if boxCollision(self.GE.player, box):
                player = self.GE.player

        for playerBullet in self.GE.playerBullets:
            if boxCollision(box, playerBullet):
                playerBullets.append(playerBullet)

        for enemy in self.GE.enemies:
            if boxCollision(box, enemy):
                enemies.append(enemy)

        for enemyBullet in self.GE.enemyBullets:
            if boxCollision(box, enemyBullet):
                enemyBullets.append(enemyBullet)

        for wall in self.GE.walls:
            if boxLineCollision(wall, box):
                walls.append(wall)

        GE = GameEntities(player, playerBullets, enemies, enemyBullets, walls)
        return GE

    def clear(self):
        for node in self.nodes:
            if len(node.nodes):
                node.clear()
        self.nodes.clear()

    def draw(self, screen):
        if not len(self.nodes):
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 1)
        else:
            for node in self.nodes:
                node.draw(screen)




