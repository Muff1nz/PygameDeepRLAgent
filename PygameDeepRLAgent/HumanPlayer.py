import pygame
import numpy as np
import WeaponSystem as ws

class HumanPlayer():
    def __init__(self, settings):
        self.type = "character"

        #Physical attributes
        self.pos = np.array([settings.screenWidth / 2, settings.screenHeight / 2])
        self.speed = 10
        self.ws = ws.WeaponSystem(settings)
        self.w = int(settings.screenWidth / 25)
        self.h = int(settings.screenHeight / 25)
        self.size = np.array([self.h, self.w])
        # needed for collision checking
        self.vertices = []
        self.vertices.append(np.array([0, 0]))
        self.vertices.append(np.array([0, self.h]))
        self.vertices.append(np.array([self.w, 0]))
        self.vertices.append(np.array([self.w, self.h]))

        #Graphical attributes
        self.sprite = pygame.image.load("./Assets/Player.png")
        self.sprite = pygame.transform.scale(self.sprite, (self.w, self.h))

    def update(self):
        keys = pygame.key.get_pressed()
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
            self.ws.shoot([0, -1], self.pos + np.array([self.w / 2, self.h / 2]))
        elif keys[pygame.K_DOWN]:
            self.ws.shoot([0, 1], self.pos + np.array([self.w / 2, self.h / 2]))
        elif keys[pygame.K_LEFT]:
            self.ws.shoot([-1, 0], self.pos + np.array([self.w / 2, self.h / 2]))
        elif keys[pygame.K_RIGHT]:
            self.ws.shoot([1, 0], self.pos + np.array([self.w / 2, self.h / 2]))

        self.ws.update()

    def draw(self, screen):
        screen.blit(self.sprite, self.pos)
        self.ws.draw(screen)