from A3CBootcampGame.Bullet import Bullet


class WeaponSystem:
    def __init__(self, settings, sprite, size=0.05):
        self.settings = settings
        self.active = False # Any active bullets?

        self.bulletCount = 50
        self.bullets = []
        for i in range(self.bulletCount):
            self.bullets.append(Bullet(settings, sprite, size))
        self.nextBullet = 0
        self.fireRate = 0.5
        self.timer = 0

    def shoot(self, dir, pos, timeStep=-1):
        if self.timer >= (self.fireRate * self.settings.gameSecond):
            self.bullets[self.nextBullet].shoot(pos, dir, timeStep)
            self.nextBullet = ((self.nextBullet + 1) % self.bulletCount)
            self.timer = 0

    def draw(self, screen):
        for bullet in self.bullets:
            if bullet.active:
                bullet.draw(screen)

    def update(self):
        self.active = False
        self.timer += 1
        for bullet in self.bullets:
            if bullet.active:
                bullet.update()
                self.active = True

    def reset(self):
        for bullet in self.bullets:
            bullet.active = False