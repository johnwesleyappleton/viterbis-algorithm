import numpy as np
from histogram_filter import HistogramFilter
from HMM_solution import HMM

'''
Test for Bayes filter
'''

# load the data
data = np.load(open('starter.npz', 'rb'))
cmap = data['arr_0']
actions = data['arr_1']
observations = data['arr_2']
belief_states = data['arr_3']

# test your code here
bayes_filter = HistogramFilter()

n, m = cmap.shape
belief = np.full((n, m), 1/(n*m))
posterior = np.zeros((len(actions), 2), dtype=int)

for i in range(len(actions)):
    belief = bayes_filter.histogram_filter(cmap, belief, actions[i], observations[i])
    belief = np.flip(belief, 0)
    argmax = np.argmax(belief)
    posterior[i, 0] = argmax % m
    posterior[i, 1] = argmax // m
    belief = np.flip(belief, 0)

print("bayes filter correct implementation: %s" % np.array_equal(belief_states[-1], posterior[-1]), "\n")


'''
Test for HMM
'''

# initialize data
observations = np.asarray([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
transition = np.asarray([[.5, .5], [.5, .5]])
emission = np.asarray([[.4, .1, .5], [.1, .5, .4]])
initial_distribution = np.full(2, .5)

# test your code here
hmm = HMM(observations, transition, emission, initial_distribution)

alpha = hmm.forward()
print("alpha:", "\n")
print(alpha, "\n")

beta = hmm.backward()
print("beta:", "\n")
print(beta, "\n")

gamma = hmm.gamma_comp(alpha, beta)

print("gamma:", "\n")
print(gamma, "\n")

print("most likely state:", "\n")
print(np.argmax(gamma, axis=1), "\n")

xi = hmm.xi_comp(alpha, beta, gamma)

print("xi:", "\n")
print(xi, "\n")

T_prime, M_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)

print("original transition:", "\n")
print(transition, "\n")

print("new transition:", "\n")
print(T_prime, "\n")

print("original emission:", "\n")
print(emission, "\n")

print("new emission:", "\n")
print(M_prime, "\n")

print("original pi:", "\n")
print(initial_distribution, "\n")

print("new pi:", "\n")
print(new_init_state, "\n")

original, prime = hmm.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)

print("original probability:", original, "\n")
print("new emission:", prime, "\n")

print("alpha:", alpha.shape == (observations.shape[0], transition.shape[0]))
print("beta:", beta.shape == (observations.shape[0], transition.shape[0]))
print("gamma:", gamma.shape == (observations.shape[0], transition.shape[0]))
print("xi:", xi.shape == (observations.shape[0]-1, transition.shape[0], transition.shape[0]))
print("transition:", T_prime.shape == transition.shape)
print("emission:", M_prime.shape == emission.shape)
print("pi:", new_init_state.shape == initial_distribution.shape)
print("original:", np.isscalar(original))
print("prime:", np.isscalar(prime))
