import numpy as np

class physicsHandler():
    def __init__(self, world, player, enemies):
        self.world = world
        # Array of boxes, could be a player, bullets, enemies...
        self.player = player
        self.enemies = enemies.enemies
        self.events = []

    # Checks for collisions between objects in game and resolves them
    def update(self, ei):
        # checks collisions with the walls
        self.boxWallCollision(self.player)
        for bullet in self.player.ws.bullets:
            if bullet.active:
                self.boxWallCollision(bullet)

        for enemy in self.enemies:
            if enemy.active or enemy.ws.active:
                self.boxWallCollision(enemy)
                for bullet in enemy.ws.bullets:
                    if bullet.active:
                        self.boxWallCollision(bullet)


        #Check if player is hitting any enemies
        for bullet in self.player.ws.bullets:
            if bullet.active:
                for enemy in self.enemies:
                    if enemy.active:
                        if self.boxCollision(bullet, enemy):
                            enemy.kill()
                            self.events.append(["Enemy killed", bullet.ei])

        #Check if enemies are hitting player
        for enemy in self.enemies:
            if enemy.ws.active:
                for bullet in enemy.ws.bullets:
                    if bullet.active:
                        if self.boxCollision(bullet, self.player):
                            self.events.append(["Player killed", ei])


    def boxWallCollision(self, box):
        for wall in self.world.walls:
            if self.boxLineCollision(wall, box):
                box.pos = box.oldPos.copy()
                if box.type == "bullet":  # Make bullets bounce off of walls
                    if wall.straight:  # Straight wall, reverse direction
                        box.dir *= -1
                    else:
                        box.dir = box.dir[::-1]  # Diagonal wall, swap x and y direction
                        if wall.start[1] > wall.end[1]:  # Descending wall, reverse after swap
                            box.dir *= -1

    # returns true if two boxes are colliding
    def boxCollision(self, box1, box2):
        if (box1.pos[1] + box1.size <= box2.pos[1] or
            box1.pos[1] >= box2.pos[1] + box2.size or
            box1.pos[0] + box1.size <= box2.pos[0] or
            box1.pos[0] >= box2.pos[0] + box2.size):
            return False
        return True

    # returns true if a box and a line is colliding
    def boxLineCollision(self, line, box):
        sign = 0
        boundsCount = 0
        for vertex in box.vertices:
            sign += np.sign(line.line(vertex[0] + box.pos[0], vertex[1] + box.pos[1]))
            if self.checkLineBounds(line, vertex + box.pos) or line.straight:
                boundsCount += 1
        if abs(sign) != 4 and boundsCount != 0:
            return True
        return False

    # returns true if a point is inside the bounds of a line
    def checkLineBounds(self, line, point):
        y = [line.start[1], line.end[1]]
        x = [line.start[0], line.end[0]]
        bounds = []
        xBound = max(x) > point[0] > min(x)
        yBound = max(y) > point[1] > min(y)
        if xBound and yBound:  # Point is inside the range of the line
            return True
        else:
            return False  # Point is outside range of line