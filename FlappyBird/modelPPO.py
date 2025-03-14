import tensorflow as tf
import numpy as np


class PPO:
    def __init__(self):
        self.numStates = 5
        self.numActions = 2
        self.discountFactor = 0.99
        self.clipEpsilon = 0.2  # best choice according to PPO paper

        initializer = tf.keras.initializers.HeNormal()
        self.actor = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.numStates,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(self.numActions, activation='softmax', kernel_initializer=initializer)])
        self.actorOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.oldActor = tf.keras.models.clone_model(self.actor)
        self.oldActor.set_weights(self.actor.get_weights())

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
        states, actions, rewards, policies, values = [], [], [], [], []
        state = env.resetGame()
        
        with tf.GradientTape(persistent=True) as tape:
            for loop in range(maxLoops):
                policy, value = self.predict(state)
                action = np.random.choice(self.numActions, p=policy.numpy())
                nextState, reward = env.gameLoop(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                policies.append(policy[action])
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

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            policies = tf.convert_to_tensor(policies, dtype=tf.float32)
            advantages = tf.convert_to_tensor(qValues, dtype=tf.float32) - tf.convert_to_tensor(values, dtype=tf.float32)

            actionsIndex = tf.range(len(actions), dtype=tf.int32)
            newProbs = tf.gather(policies, actionsIndex)
            oldPolicies = self.oldActor(states)
            self.oldActor.set_weights(self.actor.get_weights())
            oldProbs = tf.gather_nd(oldPolicies, tf.stack([actionsIndex, actions], axis=1))

            ratio = newProbs / oldProbs
            clippedRatio = tf.clip_by_value(ratio, 1 - self.clipEpsilon, 1 + self.clipEpsilon)

            self.actorLoss = -tf.reduce_mean(tf.minimum(ratio * advantages, clippedRatio * advantages))
            self.criticLoss = tf.reduce_mean(tf.square(advantages))

        actorGrads = tape.gradient(self.actorLoss, self.actor.trainable_variables)
        self.actorOptimizer.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        criticGrads = tape.gradient(self.criticLoss, self.critic.trainable_variables)
        self.criticOptimizer.apply_gradients(zip(criticGrads, self.critic.trainable_variables))

        self.score = env.score
        self.reward = sum(rewards)