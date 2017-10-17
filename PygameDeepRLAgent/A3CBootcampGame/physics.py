import numpy as np
import pygame

'''
Boxes needs to be objects that inherit from Actor

The physicsHandler will not check for collision between boxes in the same collision group,
    this is a tool to increase performance.
'''
class physicsHandler():
    def __init__(self, walls, boxes, collisionGroups, settings):
        self.settings = settings
        self.walls = walls
        self.boxes = boxes
        self.collisionGroups = collisionGroups
        self.events = []
        self.quadTree = None

    # Checks for collisions between objects in game and resolves them
    def update(self, timeStep):
        # Make quadTree
        boxes = []
        for _ in range(self.collisionGroups):
            boxes.append([])

        for box in self.boxes:
            if box.active:
                boxes[box.collisionGroup].append(box)

        self.quadTree = QuadTree(self.walls, boxes, self.collisionGroups, self.settings)
        self.traverseAndCheckQuads(self.quadTree, timeStep)

    # Recursively traverses quad tree and performs collision checking on each leaf node
    def traverseAndCheckQuads(self, quad, timeStep):
        if len(quad.nodes):
            for q in quad.nodes:
                self.traverseAndCheckQuads(q, timeStep)
        else:
            self.checkCollisions(quad.walls, quad.boxes, timeStep)

    # Collision checking performed on one node in quadtree
    def checkCollisions(self, walls, boxes, timeStep):
        # collisions for boxes
        for i, boxGroup1 in enumerate(boxes[:self.collisionGroups-1]):
            for j, boxGroup2 in enumerate(boxes[i+1:]):
                for box1 in boxGroup1:
                    for box2 in boxGroup2:
                        if (boxCollision(box1, box2)):
                            box1.onBoxCollision(box2)
                            box2.onBoxCollision(box1)
                            self.events.append({box1.type: box1.timeStep,
                                                box2.type: box2.timeStep,
                                                "timeStep": timeStep})
        for boxGroup in boxes:
            for box in boxGroup:
                if (boxWallsCollision(box, walls)):
                    box.onWallCollision()




# ============================Helper functions========================================
def boxWallsCollision(box, walls):
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

class QuadTree():
    def __init__(self, walls, boxes, collisionGroups, settings, box=Box(), depth=0):
        self.settings = settings
        self.walls = walls
        self.boxes = boxes
        self.collisionGroups = collisionGroups
        self.size = len(walls)
        for boxGroup in boxes: self.size += len(boxGroup)
        self.depth = depth
        self.box = box
        if box.size == -1:
            box.size = settings.gameRes
        self.nodes = []
        if self.size > settings.quadTreeMaxObjects and self.depth < settings.quadTreeDepth:
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
            walls, boxes = self.assignObjectsToQuad(box)
            self.nodes.append(QuadTree(walls, boxes, self.collisionGroups, self.settings, box, self.depth + 1))

    def assignObjectsToQuad(self, quad):
        walls = []
        boxes = []
        for _ in range(self.collisionGroups):
            boxes.append([])

        for wall in self.walls:
            if boxLineCollision(wall, quad):
                walls.append(wall)

        for boxGroup in self.boxes:
            for box in boxGroup:
                if boxCollision(box, quad):
                    boxes[box.collisionGroup].append(box)

        return walls, boxes

    def draw(self, screen):
        if not len(self.nodes):
            rect = pygame.Rect(self.box.pos[0], self.box.pos[1], self.box.size, self.box.size)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
        else:
            for node in self.nodes:
                node.draw(screen)




