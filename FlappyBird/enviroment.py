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
        self.reset_game()

    def reset_game(self):
            # game variables
        self.bird_y = self.SIZE[1] // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
            # pipe set up
        pygame.time.set_timer(self.ADDPIPE, 0)
        pygame.event.clear(self.ADDPIPE)
        self.add_pipe_event(first = True)

        return self.state() # (initial learning states)


    def add_pipe_event(self, first = False):
        if first:
            pygame.time.set_timer(self.ADDPIPE, self.PIPE_FREQUENCY)
        gap_y = random.randint(150, self.SIZE[1] - 150)
        top_pipe = pygame.Rect(self.SIZE[0], 0, self.PIPE_WIDTH, gap_y - self.PIPE_GAP // 2)
        bottom_pipe = pygame.Rect(
            self.SIZE[0], gap_y + self.PIPE_GAP // 2, 
            self.PIPE_WIDTH, self.SIZE[1] - (gap_y + self.PIPE_GAP // 2))
        self.pipes.append({'top': top_pipe, 'bot': bottom_pipe, 'scored': False})


    def action(self, reward):
            # move bird
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity
        bird_rect = pygame.Rect(self.BIRD_X, self.bird_y, self.BIRD_WIDTH, self.BIRD_HEIGHT)
            # move pipes 
        new_pipes = []
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
                new_pipes.append(pipe)
            # check collision
            if bird_rect.colliderect(pipe['top']) or bird_rect.colliderect(pipe['bot']):
                self.game_over = True
                reward = -5
        if self.bird_y < 0 or self.bird_y + self.BIRD_HEIGHT > self.SIZE[1]:
            self.game_over = True
            reward = -5
            # save and return
        self.pipes = new_pipes
        return reward


    def update_UI(self):
        self.screen.fill(self.BLUE)
        pygame.draw.rect(self.screen, self.WHITE, (self.BIRD_X, self.bird_y, self.BIRD_WIDTH, self.BIRD_HEIGHT))
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.GREEN, pipe['top'])
            pygame.draw.rect(self.screen, self.GREEN, pipe['bot'])
        self.screen.blit(self.font.render(f"Score: {self.score}", True, self.WHITE), (10, 10))
        pygame.display.flip()


    def state(self):
        horizontal_distance = 1000
        vertical_distance = 0
        for pipe in self.pipes:
            if pipe['top'].right >= self.BIRD_X:
                horizontal_distance = pipe['top'].x - self.BIRD_X
                vertical_distance = self.bird_y - (pipe['top'].height + self.PIPE_GAP // 2)
        state = [
            horizontal_distance / self.SIZE[0], # [0, 1]
            vertical_distance / self.SIZE[1],   # [-0.5, 0.5]
            self.bird_y / self.SIZE[1],         # [0, 1]
            self.bird_velocity / 20,            # Normalized velocity
            self.score]                         # (this state scale with reward)
        return state

    def game_loop(self, action):
        reward = 0.05
            # check actions
        if action == 1:
            self.bird_velocity = self.JUMP_VELOCITY
        reward = self.action(reward)
            # check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.ADDPIPE:
                self.add_pipe_event()
            # update game
        self.update_UI()
        self.clock.tick(self.FPS)
            # update reward

        return self.state(), reward # (learning states)