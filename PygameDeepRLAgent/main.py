import pygame
import sys
import init
import world

WHITE = 255, 255, 255

pygame.init()
settings = init.Settings()
screen = pygame.display.set_mode([settings.screen_width, settings.screen_height])

world = world.World()





def main():
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.fill(WHITE)
        world.draw(screen)
        pygame.display.flip()

if __name__ == "__main__":
    main()