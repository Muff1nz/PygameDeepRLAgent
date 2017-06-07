from init import *

def boxCollision(size1, pos1, size2, pos2):
    if (pos1[1] + size1 <= pos2[1] or
        pos1[1] >= pos2[1] + size2 or
        pos1[0] + size1 <= pos2[0] or
        pos1[0] >= pos2[0] + size2):
        return False
    return True


def checkObstacleCollision(self):
    for obstacle in obstacles:
        count = 0
        for line in obstacle.lines:
            count += 1
            sign = 0
            boundsCount = 0
            for vertex in self.vertices:
                sign += np.sign(line.line(vertex[0] + self.pos[0], vertex[1] + self.pos[1]))
                if line.checkBounds(vertex + self.pos):
                    boundsCount += 1
            if abs(sign) != 4 and boundsCount != 0:
                temp = self.dir.copy()
                self.dir[0] = temp[1]
                self.dir[1] = temp[0]
                if count == 2 or count == 4:
                    self.dir *= -1
                return True
    return False