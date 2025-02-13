import tensorflow as tf
import numpy as np


class A2C:
    def __init__(self, discount):
        self.num_states = 4
        self.num_actions = 2
        self.learning_rate = 0.001
        self.discount = discount

        initializer = tf.keras.initializers.HeNormal()
        self.actor = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(self.num_actions, activation='softmax', kernel_initializer=initializer)])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer)])
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def predict(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        policy = self.actor(state)
        value = self.critic(state)
        return tf.squeeze(policy), tf.squeeze(value)

    def training_loop(self, env, max_loops):
        states, actions, rewards, log_policies, values = [], [], [], [], []
        state = env.reset_game()
        
        with tf.GradientTape(persistent=True) as tape:
            for loop in range(max_loops):
                policy, value = self.predict(state)
                action = np.random.choice(self.num_actions, p=policy.numpy())
                next_state, reward = env.game_loop(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_policies.append(tf.math.log(policy[action] + 1e-8))
                values.append(value)

                state = next_state

                if env.game_over:
                    Qvalue = 0
                    break
                elif loop == max_loops - 1:
                    _, Qvalue = self.forward(state)
                    break

            Qvalues = np.zeros_like(rewards)
            for i in reversed(range(len(rewards))):
                Qvalue = rewards[i] + self.discount * Qvalue
                Qvalues[i] = Qvalue

            log_policies = tf.convert_to_tensor(log_policies, dtype=tf.float32)
            advantages = tf.convert_to_tensor(Qvalues, dtype=tf.float32) - tf.convert_to_tensor(values, dtype=tf.float32)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            actor_loss = -tf.reduce_mean(log_policies * advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.score = env.score
        self.reward = sum(rewards)
        states, actions, rewards, log_policies, values = [], [], [], [], []
        state = env.reset_game()
