import pygame
import tensorflow as tf
from envBird import FlappyBird
from modelBird import A2Cmodel


class testBird:
    def __init__(self, mode):
        self.game = FlappyBird()
        if mode == "Y":
            self.human()
        else:
            self.AI()

    def human(self):
        while not self.game.game_over:
            action = 0
            if pygame.event.peek(pygame.KEYDOWN):
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: 
                    action = 1
            state, reward = self.game.game_loop(action)
            # print(states, reward)
            if self.game.game_over:
                self.game.reset_game()
        pygame.quit()

    def AI(self):
        model = A2Cmodel(4, 2, 0.999)
        model.actor = tf.keras.models.load_model('5flappybirdA2C/actor.keras')
        model.critic = tf.keras.models.load_model('5flappybirdA2C/critic.keras')

        for i in range(1000):
            try:
                model.training_loop(self.game, 10000)
            except KeyboardInterrupt:
                pygame.quit()


if __name__ == "__main__":
    while True:
        mode = input("I'm not a robot: Y/N ").strip().upper()
        if mode in ["Y", "N"]:
            break
    testBird(mode)
    