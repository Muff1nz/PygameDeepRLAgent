import pygame
import sys

import init
import world
import physics
import HumanPlayer as hp

WHITE = 255, 255, 255

pygame.init()
settings = init.Settings()
screen = pygame.display.set_mode([settings.screenWidth, settings.screenHeight])

world = world.World(settings)

player = hp.HumanPlayer(settings)

physics = physics.physicsHandler(world, player)

def main():
    fpsTimer = 0
    while 1:
        if (fpsTimer + settings.mspf) <= pygame.time.get_ticks():
            fpsTimer = pygame.time.get_ticks()

            #Check events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()


            #Update stuff
            player.update()
            physics.update()

            #Render stuff
            screen.fill(WHITE)
            world.draw(screen)
            player.draw(screen)
            pygame.display.flip()

if __name__ == "__main__":
    main()