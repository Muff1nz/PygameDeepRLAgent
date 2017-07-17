import pygame
import sys
import tensorflow as tf

from init import Settings
from world import World
from physics import physicsHandler
from HumanPlayer import HumanPlayer
from DQNAgent import DQNAgent
from EnemyHandler import EnemyHandler
from GameHandler import GameHandler
from ReplayMemory import ReplayMemory

WHITE = 255, 255, 255

pygame.init()
settings = Settings()
screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])

world = World(settings)

player = DQNAgent(settings, "./Assets/Player.png")
enemyHandler = EnemyHandler(settings, "./Assets/Enemy.png", world)

physics = physicsHandler(world, player, enemyHandler)

gameHandler = GameHandler(physics.events, enemyHandler, player)
replayMemory = ReplayMemory(settings)

def main():
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(settings.tbPath)
        fpsTimer = 0
        time = 0
        frames = 0
        while 1:
            if (fpsTimer + settings.mspf) <= pygame.time.get_ticks():
                fpsTimer = pygame.time.get_ticks()
                if time + 1025 < pygame.time.get_ticks():
                    print("fps: " + str(frames))
                    print("Player score: "  + str(gameHandler.playerScore))
                    time = pygame.time.get_ticks()
                    frames = 0

                #Check events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        writer.close()
                        sys.exit()


                #Update stuff
                enemyHandler.update()
                player.update(replayMemory.ei)
                physics.update(replayMemory.ei)
                gameHandler.update(replayMemory)

                #Render stuff
                screen.fill(WHITE)
                world.draw(screen)
                enemyHandler.draw(screen)
                player.draw(screen)
                pygame.display.flip()
                frames += 1

                #Replay Memory
                replayMemory.update(screen, player)

                # log images, for testing
                if not frames % settings.deepRLSampleRate and settings.logProcessedFrames:
                    image = replayMemory.processedFrames[replayMemory.ei]
                    imageSummary = tf.expand_dims(image, 0)
                    imageSummary = tf.expand_dims(imageSummary, 3)
                    imageSummary = tf.summary.image("pFrame" + str(settings.version), imageSummary)
                    imageSummary = sess.run(imageSummary)
                    writer.add_summary(imageSummary)

if __name__ == "__main__":
    main()