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

def main():
    pygame.init()
    settings = Settings()
    screen = pygame.display.set_mode([settings.screenRes, settings.screenRes])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = settings.gpuMemoryFraction
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(settings.tbPath)
        world = World(settings)
        replayMemory = ReplayMemory(settings)
        player = DQNAgent(settings, "./Assets/Player.png", sess, replayMemory, writer)
        enemyHandler = EnemyHandler(settings, "./Assets/Enemy.png", world)
        physics = physicsHandler(world, player, enemyHandler, settings)
        gameHandler = GameHandler(physics.events, enemyHandler, player, writer, sess)

        fpsTimer = 0
        time = 0
        frames = 0
        counter = 0

        sess.run(tf.global_variables_initializer())
        while 1:
            if (fpsTimer + settings.mspf) <= pygame.time.get_ticks():
                fpsTimer = pygame.time.get_ticks()
                if time + 1000 < pygame.time.get_ticks():
                    print("================================================")
                    print("fps: " + str(frames))
                    print("Player score: " + str(gameHandler.playerScore))
                    print("Collision checks: " + str(physics.collisionChecks))
                    print("dqn global step: " + str(sess.run(player.step)))
                    print("ei: " + str(replayMemory.ei))
                    print("================================================")
                    time = pygame.time.get_ticks()
                    frames = 0

                #Check events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if settings.saveCheckpoint:
                            player.save()
                        replayMemory.close()
                        writer.close()
                        sys.exit()


                #Update stuff
                enemyHandler.update()
                player.update(replayMemory.ei, counter)
                physics.update(replayMemory.ei)
                gameHandler.update(replayMemory)

                #Render stuff
                screen.fill(WHITE)
                world.draw(screen)
                enemyHandler.draw(screen)
                player.draw(screen)
                if not (frames % settings.deepRLRate):
                    replayMemory.update(screen, player)
                if settings.renderQuads:
                    physics.quadTree.draw(screen)
                pygame.display.flip()
                frames += 1

                # log images, for testing
                if not (counter % settings.deepRLRate):
                    if settings.logProcessedFrames:
                        imageSummary = [replayMemory.getState(settings.experienceMemorySize-1)]
                        imageSummary = tf.expand_dims(imageSummary[0], 3)
                        imageSummary = tf.summary.image("pFrame" + str(settings.version), imageSummary, max_outputs=4)
                        imageSummary = sess.run(imageSummary)
                        writer.add_summary(imageSummary)
                counter += 1

if __name__ == "__main__":
    main()