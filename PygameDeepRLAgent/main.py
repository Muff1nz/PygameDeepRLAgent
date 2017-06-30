import pygame
import sys

import init
import world
import physics
import HumanPlayer as hp
import EnemyHandler as em
import GameHandler as gh

WHITE = 255, 255, 255

pygame.init()
settings = init.Settings()
screen = pygame.display.set_mode([settings.screenWidth, settings.screenHeight])

world = world.World(settings)

player = hp.HumanPlayer(settings)
enemyHandler = em.EnemyHandler(settings, world)

physics = physics.physicsHandler(world, player, enemyHandler)

gameHandler = gh.GameHandler(physics.events, enemyHandler, player)

def main():
    fpsTimer = 0
    time = 0
    frames = 0
    while 1:
        if (fpsTimer + settings.mspf) <= pygame.time.get_ticks():
            fpsTimer = pygame.time.get_ticks()
            if time + 1000 < pygame.time.get_ticks():
                print("fps: " + str(frames))
                print("Player score: "  + str(gameHandler.playerScore))
                time = pygame.time.get_ticks()
                frames = 0

            #Check events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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

if __name__ == "__main__":
    main()