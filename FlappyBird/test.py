import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pygame
import tensorflow as tf
from enviroment import FlappyBird
from modelA2C import A2C
from modelPPO import PPO


class TestBird:
    def __init__(self, mode):
        self.game = FlappyBird()
        if mode == "Y":
            self.human()
        else:
            model = input("Choose model numbers: 1[A2C] 2[PPO]").strip()
            self.ai(model)

    def human(self):
        while not self.game.gameOver:
            action = 0
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_UP:
                    action = 1
            self.game.gameLoop(action)
            if self.game.gameOver:
                self.game.resetGame()
        pygame.quit()

    def ai(self, model):
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
                model.trainingLoop(self.game, 10000)
            except KeyboardInterrupt:
                pygame.quit()


if __name__ == "__main__":
    while True:
        mode = input("I'm not a robot: Y/N ").strip().upper()
        if mode in ["Y", "N"]:
            break
    TestBird(mode)
    