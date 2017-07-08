import pygame
import sys
import tensorflow as tf

import init
import world
import physics
import HumanPlayer as hp
import EnemyHandler as em
import GameHandler as gh
import ReplayMemory as rm

WHITE = 255, 255, 255

pygame.init()
settings = init.Settings()
screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])

world = world.World(settings)

player = hp.HumanPlayer(settings)
enemyHandler = em.EnemyHandler(settings, world)

physics = physics.physicsHandler(world, player, enemyHandler)

gameHandler = gh.GameHandler(physics.events, enemyHandler, player)
replayMemory = rm.ReplayMemory(settings)

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
                player.update()
                physics.update()
                gameHandler.update()

                #Render stuff
                screen.fill(WHITE)
                world.draw(screen)
                enemyHandler.draw(screen)
                player.draw(screen)
                pygame.display.flip()
                frames += 1

                #Replay Memory
                replayMemory.update(screen)

                # log images, for testing
                if not frames % settings.deepRLSampleRate:
                    image = replayMemory.processedFrames[replayMemory.ei]
                    imageSummary = tf.expand_dims(image, 0)
                    imageSummary = tf.expand_dims(imageSummary, 3)
                    imageSummary = tf.summary.image("pFrame" + str(settings.version), imageSummary)
                    imageSummary = sess.run(imageSummary)
                    writer.add_summary(imageSummary)

if __name__ == "__main__":
    main()