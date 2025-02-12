import tensorflow as tf
import numpy as np


class A2C:
    def __init__(self, discount, clip_epsilon, minibatch_size, minibatch_epoch):
        self.num_states = 4
        self.num_actions = 2
        self.learning_rate = 0.001
        self.discount = discount
        self.clip_epsilon = clip_epsilon
        self.minibatch_size = minibatch_size
        self.minibatch_epoch = minibatch_epoch

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

    def train(self, states, actions, rewards, old_policies, values):
        Qvalues = np.zeros_like(rewards)
        Qvalue = 0
        for i in reversed(range(len(rewards))):
            Qvalue = rewards[i] + self.discount * Qvalue
            Qvalues[i] = Qvalue

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        old_policies = tf.convert_to_tensor(old_policies)
        advantages = tf.convert_to_tensor(Qvalues) - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        with tf.GradientTape(persistent=True) as tape:
            for _ in range(self.minibatch_epoch):
                new_policy = self.actor(states)
                value = self.critic(states)

                ratio = new_policy/(old_policies)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                critic_loss = tf.reduce_mean(tf.square(advantages))
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def training_loop(self, env, max_loops):
        states, actions, rewards, old_policies, values = [], [], [], [], []
        state = env.reset_game()

        for loop in range (max_loops):
            policy, value = self.predict(state)
            action = np.random.choice(self.num_actions, p=policy.numpy())
            next_state, reward = env.game_loop(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_policies.append(policy[action])
            values.append(value)

            state = next_state
            self.score = env.score
            self.reward = sum(rewards)
            if env.game_over or loop == max_loops - 1:
                self.train(states, actions, rewards, old_policies, values)
                states, actions, rewards, old_policies, values = [], [], [], [], []
                state = env.reset_game()