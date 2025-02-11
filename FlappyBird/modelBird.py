import tensorflow as tf
import numpy as np


class A2Cmodel:
    def __init__(self, num_states, num_actions, discount):
            # hyperparameters
        initializer = tf.keras.initializers.HeNormal()
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
            # actor model
        self.actor = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(num_actions, activation='softmax', kernel_initializer=initializer)])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            # critic model
        self.critic = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.num_states,)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer)])
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def predict(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        policy = self.actor(state)
        value = self.critic(state)
        return tf.squeeze(policy), tf.squeeze(value)

    def training_loop(self, env, max_loops):
            # reinforcement learning components
        state = env.reset_game() # (initial learning states)
        rewards = []
        log_policies = []
        values = []

        with tf.GradientTape(persistent = True) as tape:
            for loop in range (max_loops):
                    # action
                policy, value = self.predict(state)
                action = np.random.choice(self.num_actions, p=policy.numpy())
                state, reward = env.game_loop(action) # (learning states)
                    # update components
                rewards.append(reward)
                log_policy = tf.math.log(policy[action] + 1e-8)
                log_policies.append(log_policy)
                values.append(value)
                    # check game_loop/training_loop end conditions
                if env.game_over:
                    Qvalue = 0
                    break
                elif loop == max_loops - 1:
                    _, Qvalue = self.predict(state)
                    break
                # calculate Q(Bellman equation)
            Qvalues = np.zeros_like(values)
            for i in reversed(range(len(values))):
                Qvalue = rewards[i] + self.discount * Qvalue
                Qvalues[i] = Qvalue
            Qvalues = (Qvalues - np.mean(Qvalues)) / (np.std(Qvalues) + 10e-8) # normalizing
                # calculate advantages
            advantages = []
            for Qvalue, value in zip(Qvalues, values):
                advantages.append(Qvalue - value)
                # calculate losses
            actor_loss = 0
            for log_policy, advantage in zip(log_policies, advantages):
                actor_loss -= (log_policy * advantage) / len(log_policies)
            critic_loss = 0.5 * tf.math.reduce_mean(tf.math.square(advantages))
            # apply gradients
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.score = env.score
        self.reward = sum(rewards)