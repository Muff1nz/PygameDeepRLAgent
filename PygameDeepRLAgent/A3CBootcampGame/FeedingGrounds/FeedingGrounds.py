import asyncio
import numpy as np
import pygame

from A3CBootcampGame.FeedingGrounds.Food import FoodHandler
from A3CBootcampGame.FeedingGrounds.GameHandler import GameHandler
from A3CBootcampGame.FeedingGrounds.Player import Player
from A3CBootcampGame.FeedingGrounds.world import World
from A3CBootcampGame.physics import physicsHandler

WHITE = 255, 255, 255
# Class for the feeding grounds level in A3CBootCamp
class FeedingGrounds():
    def __init__(self, settings, gameDataQueue, playerActionQueue):
        self.settings = settings
        self.gameDataQueue = gameDataQueue
        self.playerActionQueue = playerActionQueue
        self.initGame()

    def initGame(self):
        initSettings = self.playerActionQueue.get()
        self.window = initSettings["WindowSettings"]
        self.worker = initSettings["worker"]

        if self.window:
            pygame.display.init()
            self.screen = pygame.display.set_mode([self.settings.screenRes, self.settings.screenRes])
        self.gameScreen = pygame.Surface((self.settings.gameRes, self.settings.gameRes))

        self.runGame = True
        self.episodeInProgress = True
        self.frames = []
        self.episodeData = []

        self.gameCounter = 0

        self.playerAction = 0
        self.playerTimeStep = -1

        self.world = World(self.settings)
        self.player = Player(self.settings, "./Assets/Player.png")
        self.foodHandler = FoodHandler(self.settings, self.player)
        self.gameHandler = GameHandler(self.player, self.foodHandler)

        collisionGroups = 2
        boxes = []
        self.player.collisionGroup = 0
        boxes.append(self.player)
        for foodBit in self.foodHandler.food:
            foodBit.collisionGroup = 1
            boxes.append(foodBit)

        self.physics = physicsHandler(self.world.walls, boxes,  collisionGroups, self.settings)

    async def run(self):
        while True:
            if not self.episodeInProgress:
                # send data to worker
                self.gameDataQueue.put([self.worker,
                                       ["EpisodeData", np.array(self.episodeData), self.gameHandler.playerScore]])
                # reset game
                self.episodeData = []
                self.frames = []
                self.gameHandler.resetGame()
                self.playerTimeStep = -1
                self.bootStrapCounter = 0
                self.gameCounter = 0

            # Render stuff
            self.gameScreen.fill(WHITE)
            self.world.draw(self.gameScreen)
            self.foodHandler.draw(self.gameScreen)
            self.player.draw(self.gameScreen)
            if self.window:
                if pygame.mouse.get_focused():
                    if self.settings.renderQuads:
                        self.physics.quadTree.draw(self.gameScreen)
                    montiroFrame = pygame.transform.scale(self.gameScreen.copy(),
                                                          (self.settings.screenRes, self.settings.screenRes))
                    self.screen.blit(montiroFrame, montiroFrame.get_rect())
                    pygame.display.flip()

            if not self.gameCounter % self.settings.deepRLRate:
                frame = pygame.surfarray.array3d(self.gameScreen.copy())
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
                # Send frame to agent
                self.gameDataQueue.put([self.worker, ["CurrentFrame", frame]])
                while self.playerActionQueue.empty(): # Yield to let other games run, to prevent blocking on the queue
                    await asyncio.sleep(0.005)
                self.playerAction = self.playerActionQueue.get()
                self.playerTimeStep += 1
                # Rewards default to 0, game handler will track causality and update
                self.episodeData.append([frame, self.playerAction, 0])

            # Update stuff
            self.player.update(self.playerAction)
            self.physics.update(self.playerTimeStep)
            self.foodHandler.update(self.gameCounter)
            self.episodeInProgress = self.gameHandler.update(self.physics.events, self.gameCounter, self.episodeData)

            if self.window:
                # Check events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.gameDataQueue.put(["Game closed!"])
            self.gameCounter += 1