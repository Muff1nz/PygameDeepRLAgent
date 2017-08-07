import pygame
import sys

from EnemyHandler import EnemyHandler
from GameHandler import GameHandler
from RandomAgent import RandomAgent
from physics import physicsHandler
from world import World

WHITE = 255, 255, 255
class ClusterCube:
    def __init__(self, settings):
        pygame.init()
        self.settings = settings
        self.screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])

        self.world = World(settings)
        self.player = RandomAgent(settings, "./Assets/Player.png")
        self.enemyHandler = EnemyHandler(settings, "./Assets/Enemy.png", self.world)
        self.physics = physicsHandler(self.world, self.player, self.enemyHandler, self.settings)
        self.gameHandler = GameHandler(self.physics.events, self.enemyHandler, self.player)

    def runGameLoop(self):
        # Update stuff
        self.enemyHandler.update()
        self.player.update()
        self.physics.update(0)
        self.gameHandler.update()

        # Render stuff
        self.screen.fill(WHITE)
        self.world.draw(self.screen)
        self.enemyHandler.draw(self.screen)
        self.player.draw(self.screen)
        if self.settings.renderQuads:
            self.physics.quadTree.draw(self.screen)
        pygame.display.flip()

        # Check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()