import numpy as np
import pygame


class physicsHandler():
    def __init__(self, world, player, enemies, settings):
        self.settings = settings
        self.world = world
        self.player = player
        self.enemies = enemies
        self.events = []
        self.quadTree = None

        self.pc = 0

    # Checks for collisions between objects in game and resolves them
    def update(self, playerTimeStep):
        # Make quadTree
        playerBullets = []
        enemyBullets = []
        enemies = []

        if self.player.ws.active:
            for bullet in self.player.ws.bullets:
                if bullet.active:
                    playerBullets.append(bullet)

        for enemy in self.enemies:
            if enemy.active:
                enemies.append(enemy)
            if enemy.ws.active:
                for bullet in enemy.ws.bullets:
                    if bullet.active:
                        enemyBullets.append(bullet)

        GE = GameEntities(self.player, playerBullets, enemies, enemyBullets, self.world.walls)
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
            for bullet in GE.playerBullets:
                for enemy in GE.enemies:
                    if boxCollision(bullet, enemy):
                        enemy.kill()
                        bullet.onCollision()
                        if self.settings.causalityTracking:
                            self.events.append(["Player hit enemy!", bullet.playerTimeStep])
                        else:
                            self.events.append(["Player hit enemy!", playerTimeStep])
                if boxWallCollision(bullet, GE.walls):
                    bullet.onCollision()

        # collisions for enemies
        for enemy in GE.enemies:
            if boxWallCollision(enemy, GE.walls):
                enemy.onWallCollision()
            for bullet in GE.enemyBullets:
                if GE.player:
                    if boxCollision(bullet, GE.player):
                        self.events.append(["Enemy hit player!", playerTimeStep])
                if boxWallCollision(bullet, GE.walls):
                    bullet.onCollision()


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
    for vertex in box.vertices:
        sign += np.sign(line.line(vertex[0] + box.pos[0], vertex[1] + box.pos[1]))
    if abs(sign) != 4:
        return True
    return False


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
    def __init__(self, player, playerBullets, enemies, enemyBullets, walls):
        self.player = player
        self.playerBullets = playerBullets
        self.enemies = enemies
        self.enemyBullets = enemyBullets
        self.walls = walls
        self.count = len(self.playerBullets) + len(self.enemies) + len(self.enemyBullets) + len(walls)
        if player:
            self.count += 1


class QuadTree():
    def __init__(self, GE, settings, box=Box(), depth=0):
        self.settings = settings
        self.GE = GE
        self.depth = depth
        self.box = box
        if box.size == -1:
            box.size = settings.gameRes
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
        playerBullets = []
        enemies = []
        enemyBullets = []
        walls = []

        if self.GE.player:
            if boxCollision(self.GE.player, box):
                player = self.GE.player

        for bullet in self.GE.playerBullets:
            if boxCollision(box, bullet):
                playerBullets.append(bullet)

        for enemy in self.GE.enemies:
            if boxCollision(enemy, box):
                    enemies.append(enemy)

        for bullet in self.GE.enemyBullets:
            if boxCollision(box, bullet):
                enemyBullets.append(bullet)

        for wall in self.GE.walls:
            if boxLineCollision(wall, box):
                walls.append(wall)

        GE = GameEntities(player, playerBullets, enemies, enemyBullets, walls)
        return GE

    def draw(self, screen):
        if not len(self.nodes):
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 1)
        else:
            for node in self.nodes:
                node.draw(screen)




