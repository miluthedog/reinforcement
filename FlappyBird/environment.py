import pygame
import random
import sys


class FlappyBird:
    def __init__(self):
        pygame.init()
        self.WHITE = (255, 255, 255)
        self.BLUE = (135, 206, 235)
        self.GREEN = (0, 200, 0)
        # game constants
        self.SIZE = (400, 600)
        self.FPS = 60
        # bird constants
        self.BIRD_WIDTH = 30
        self.BIRD_HEIGHT = 30
        self.BIRD_X = self.SIZE[0] // 5
        self.JUMP_VELOCITY = -10
        self.GRAVITY = 0.5
        # pipes constants
        self.PIPE_WIDTH = 60
        self.PIPE_GAP = 180
        self.PIPE_VELOCITY = 3
        self.PIPE_FREQUENCY = 1500
        self.ADDPIPE = pygame.USEREVENT + 1
        # game set up
        self.screen = pygame.display.set_mode(self.SIZE)
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.resetGame()

    def resetGame(self):
        # game variables
        self.birdY = self.SIZE[1] // 2
        self.birdVelocity = 0
        self.pipes = []
        self.score = 0
        self.gameOver = False
        # pipe set up
        pygame.time.set_timer(self.ADDPIPE, 0)
        pygame.event.clear(self.ADDPIPE)
        self.addPipeEvent(first = True)

        return self.getState() # (initial learning states)

    def addPipeEvent(self, first = False):
        if first:
            pygame.time.set_timer(self.ADDPIPE, self.PIPE_FREQUENCY)
        gapY = random.randint(150, self.SIZE[1] - 150)
        topPipe = pygame.Rect(self.SIZE[0], 0, self.PIPE_WIDTH, gapY - self.PIPE_GAP // 2)
        bottomPipe = pygame.Rect(
            self.SIZE[0], gapY + self.PIPE_GAP // 2, 
            self.PIPE_WIDTH, self.SIZE[1] - (gapY + self.PIPE_GAP // 2))
        self.pipes.append({'top': topPipe, 'bot': bottomPipe, 'scored': False})

    def updateBird(self, action):
        # move bird
        if action == 1:
            self.birdVelocity = self.JUMP_VELOCITY
        self.birdVelocity += self.GRAVITY
        self.birdY += self.birdVelocity
        birdRect = pygame.Rect(self.BIRD_X, self.birdY, self.BIRD_WIDTH, self.BIRD_HEIGHT)
        # move pipes 
        newPipes = []
        reward = 0.05

        for pipe in self.pipes:
            pipe['top'].x -= self.PIPE_VELOCITY
            pipe['bot'].x -= self.PIPE_VELOCITY
            # score pipe
            if pipe['top'].right < self.BIRD_X and not pipe['scored']:
                self.score += 1
                reward = 5
                pipe['scored'] = True
            # remove pipe
            if pipe['top'].right > 0:
                newPipes.append(pipe)
            # check collision
            if birdRect.colliderect(pipe['top']) or birdRect.colliderect(pipe['bot']):
                self.gameOver = True
                reward = -5
        if self.birdY < 0 or self.birdY + self.BIRD_HEIGHT > self.SIZE[1]:
            self.gameOver = True
            reward = -5
        # save and return
        self.pipes = newPipes
        return reward

    def updateUI(self):
        self.screen.fill(self.BLUE)
        pygame.draw.rect(self.screen, self.WHITE, (self.BIRD_X, self.birdY, self.BIRD_WIDTH, self.BIRD_HEIGHT))
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.GREEN, pipe['top'])
            pygame.draw.rect(self.screen, self.GREEN, pipe['bot'])
        self.screen.blit(self.font.render(f"Score: {self.score}", True, self.WHITE), (10, 10))
        pygame.display.flip()

    def getState(self):
        horizontalDistance = 1000
        verticalDistance = 0
        for pipe in self.pipes:
            if pipe['top'].right >= self.BIRD_X:
                horizontalDistance = pipe['top'].x - self.BIRD_X
                verticalDistance = self.birdY - (pipe['top'].height + self.PIPE_GAP // 2)
        state = [
            horizontalDistance / self.SIZE[0], # [0, 1]
            verticalDistance / self.SIZE[1],   # [-0.5, 0.5]
            self.birdY / self.SIZE[1],         # [0, 1]
            self.birdVelocity / 20,            # Normalized velocity
            self.score]                         # (this state scale with reward)
        return state

    def gameLoop(self, action):
        reward = self.updateBird(action)
        # check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.ADDPIPE:
                self.addPipeEvent()
        # update game
        self.updateUI()
        self.clock.tick(self.FPS)
        # update reward

        return self.getState(), reward # (learning states)