import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        alpha = np.zeros((len(self.Observations), len(self.Initial_distribution)))
        alpha[0] = self.Emission[:, self.Observations[0]].T * self.Initial_distribution
        for i in range(1, len(self.Observations)):
            alpha[i] = self.Emission[:, self.Observations[i]].T * (alpha[i-1] @ self.Transition)

        return alpha

    def backward(self):
        beta = np.ones((len(self.Observations), len(self.Initial_distribution)))
        for i in range(2, len(self.Observations) + 1):
            beta[-i] = (beta[-i+1] * self.Emission[:, self.Observations[-i+1]].T) @ self.Transition.T

        return beta

    def gamma_comp(self, alpha, beta):
        gamma = alpha * beta / np.sum(alpha[-1])

        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.zeros((len(self.Observations) - 1, len(self.Transition), len(self.Transition)))
        for i in range(len(self.Observations) - 1):
            xi[i] = self.Emission[:, self.Observations[i+1]] * (alpha[i] * self.Transition.T).T * beta[i+1]
            xi[i] /= np.sum(xi[i])

        return xi

    def update(self, alpha, beta, gamma, xi):
        new_init_state = gamma[0]
        T_prime = (np.sum(xi, axis=0).T / np.sum(gamma[:-1], axis=0)).T
        obs = np.zeros((len(self.Observations), self.Emission.shape[1]))
        for i in range(self.Emission.shape[1]):
            obs[:, i] = self.Observations == i
        M_prime = ((gamma.T @ obs).T / np.sum(gamma, axis=0)).T

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        new_beta = np.ones_like(beta)
        for i in range(2, len(self.Observations) + 1):
            new_beta[-i] = (new_beta[-i+1] * M_prime[:, self.Observations[-i+1]].T) @ T_prime.T

        p_old = beta[0] * self.Initial_distribution @ self.Emission[:, self.Observations[0]]
        p_new = new_beta[0] * new_init_state @ M_prime[:, self.Observations[0]]

        return p_old, p_new
