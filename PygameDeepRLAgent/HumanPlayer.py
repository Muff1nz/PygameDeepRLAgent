import pygame
import numpy as np
import WeaponSystem as ws

class HumanPlayer():
    def __init__(self, settings):
        self.type = "character"
        self.settings = settings

        #Physical attributes
        self.pos = np.array([settings.screenRes / 2, settings.screenRes / 5])
        self.oldPos = self.pos.copy()
        self.speed = 10
        self.ws = ws.WeaponSystem(settings, "./Assets/Player.png")
        self.size = settings.screenRes // 25
        # needed for collision checking
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.size]))
        self.vertices.append(np.array([self.size, 0]))
        self.vertices.append(np.array([self.size, self.size]))

        #Graphical attributes
        self.sprite = pygame.image.load("./Assets/Player.png")
        self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))

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

















