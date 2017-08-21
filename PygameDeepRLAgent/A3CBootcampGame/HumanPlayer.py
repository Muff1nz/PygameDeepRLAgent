import pygame
import numpy as np

from Actor import Actor

class HumanPlayer(Actor):
    def __init__(self, settings, spritePath):
        super(HumanPlayer, self).__init__(settings, spritePath)
        self.type = "player"

    def update(self):
        keys = pygame.key.get_pressed()
        self.oldPos = self.pos.copy()
        # Movement
        if keys[pygame.K_d]:
            self.pos[0] += self.speed
        elif keys[pygame.K_a]:
            self.pos[0] -= self.speed
        elif keys[pygame.K_w]:
            self.pos[1] -= self.speed
        elif keys[pygame.K_s]:
            self.pos[1] += self.speed

        # Bullets, +self.size/2 to get the bullet centered
        if keys[pygame.K_UP]:
            self.ws.shoot([0, -1], self.pos + np.array([self.size / 2, self.size / 2]))
        elif keys[pygame.K_DOWN]:
            self.ws.shoot([0, 1], self.pos + np.array([self.size / 2, self.size / 2]))
        elif keys[pygame.K_LEFT]:
            self.ws.shoot([-1, 0], self.pos + np.array([self.size / 2, self.size / 2]))
        elif keys[pygame.K_RIGHT]:
            self.ws.shoot([1, 0], self.pos + np.array([self.size / 2, self.size / 2]))

        self.ws.update()

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)

    def reset(self):
        self.pos = np.array([self.settings.screenRes / 2, self.settings.screenRes / 5])
        self.ws.reset()

















