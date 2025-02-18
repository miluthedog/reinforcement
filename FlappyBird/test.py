import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pygame
import tensorflow as tf
from enviroment import FlappyBird
from modelA2C import A2C
from modelPPO import PPO


class testBird:
    def __init__(self, mode):
        self.game = FlappyBird()
        if mode == "Y":
            self.human()
        else:
            model = input("Choose model numbers: 1[A2C] 2[PPO]").strip()
            self.AI(model)

    def human(self):
        while not self.game.game_over:
            action = 0
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_UP:
                    action = 1
            self.game.game_loop(action)
            if self.game.game_over:
                self.game.reset_game()
        pygame.quit()


    def AI(self, model):
        if model == "1":
            model = A2C()
            model.actor = tf.keras.models.load_model('FlappyBird/models/A2Cactor.keras')
            model.critic = tf.keras.models.load_model('FlappyBird/models/A2Ccritic.keras')
        if model == "2":
            model = PPO()
            model.actor = tf.keras.models.load_model('FlappyBird/models/PPOactor.keras')
            model.critic = tf.keras.models.load_model('FlappyBird/models/PPOcritic.keras')           

        for _ in range(1000):
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
    