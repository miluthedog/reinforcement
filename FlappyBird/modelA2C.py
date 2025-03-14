import tensorflow as tf
import numpy as np


class A2C:
    def __init__(self):
        self.numStates = 5
        self.numActions = 2
        self.discountFactor = 0.999

        initializer = tf.keras.initializers.HeNormal()
        self.actor = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.numStates,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(self.numActions, activation='softmax', kernel_initializer=initializer)])
        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.numStates,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer)])
        self.criticOptimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)


    def predict(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        policy = self.actor(state)
        value = self.critic(state)
        return tf.squeeze(policy), tf.squeeze(value)

    def trainingLoop(self, env, maxLoops):
        states, actions, rewards, logPolicies, values = [], [], [], [], []
        state = env.resetGame()
        
        with tf.GradientTape(persistent=True) as tape:
            for loop in range(maxLoops):
                policy, value = self.predict(state)
                action = np.random.choice(self.numActions, p=policy.numpy())
                nextState, reward = env.gameLoop(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                logPolicies.append(tf.math.log(policy[action] + 1e-8))
                values.append(value)

                state = nextState

                if env.gameOver:
                    qValue = 0
                    break
                elif loop == maxLoops - 1:
                    _, qValue = self.predict(state)
                    break

            qValues = np.zeros_like(rewards)
            for i in reversed(range(len(rewards))):
                qValue = rewards[i] + self.discountFactor * qValue
                qValues[i] = qValue

            logPolicies = tf.convert_to_tensor(logPolicies, dtype=tf.float32)
            advantages = tf.convert_to_tensor(qValues, dtype=tf.float32) - tf.convert_to_tensor(values, dtype=tf.float32)
 
            self.actorLoss = -tf.reduce_mean(logPolicies * advantages)
            self.criticLoss = tf.reduce_mean(tf.square(advantages))

        actorGrads = tape.gradient(self.actorLoss, self.actor.trainable_variables)
        self.actorOptimizer.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        criticGrads = tape.gradient(self.criticLoss, self.critic.trainable_variables)
        self.criticOptimizer.apply_gradients(zip(criticGrads, self.critic.trainable_variables))

        self.score = env.score
        self.reward = sum(rewards)