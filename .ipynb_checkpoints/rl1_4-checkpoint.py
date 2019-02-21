class REINFORCE(object):

  def __init__(
      self, number_of_arms, step_size=0.1, baseline=False, temperature=1.0):
    self._number_of_arms = number_of_arms
    self._lr = step_size
    self._baseline = baseline
    self._temperature = temperature
    self.name = 'reinforce, baseline: {}, temperature: {}'.format(
        baseline, temperature)
    self.reset()

#   def policy(self, *args):
#     T = self._temperature
#     if len(args) == 1:
#       h_a = self._estimates[args]
#       h_bs = np.sum(self._estimates[np.arange(self._number_of_arms) != h_a])
#     else:
#       h_a = self._estimates[args[0]:args[len(args)]]
#       h_bs = [np.sum([args[j] for j in range(len(args)) if j != i]) for i in range(len(args))]
      
#     p_a = np.divide(np.exp(h_a / T), np.exp(h_bs / T))
#     return p_a
  
  def get_baseline(self):
    if self._baseline:
      t = np.sum(self._counts)
      return self.R_t / t
    
    return self._baseline
  
#   def update(self, *action, reward, selected=True):
#     H_t = self._estimates[action]
#     b = self.get_baseline()
#     r = reward
#     if selected:
#       G_th = (r - b) * (1 - policy(action))
#       return H_t + self._lr * G_th
#     else:
#       G_th = (r - b) * policy(action)
#       return H_t - self._lr * G_th
    
  def step(self, previous_action, reward):
    T = self._temperature
    if previous_action is not None:
      r = reward
      self._counts[previous_action] += 1.0
      self.R_t += r
      b = self.get_baseline()
      self._estimates -= (self._lr * (r - b) * self._policy) / T
      self._estimates[previous_action] += (self._lr * (r - b)) / T
#       unselected_actions = np.concatenate([range(previous_action)], [range(previous_action + 1, self._number_of_arms)])
      
      phi = (self._estimates - self._estimates[previous_action])
      self._policy = np.divide(np.exp(phi / T), np.sum(np.exp(phi / T)))
      probs = np.exp(self._estimates) / np.exp(self._estimates).sum()
      a = np.random.choice(probs, p=probs)
      action = np.argmax(self._estimates == a)
      return action
      
    action = np.random.randint(self._number_of_arms)
    return action

  def reset(self):
    """
    Resets all the statistics (estimated rewards and counts) to zero.
    
    This is also used to initialise these statistics on initialisation.
    """
    self.R_t = 0
    self._policy = np.ones((self._number_of_arms,)) / self._number_of_arms
    self._estimates = np.zeros((self._number_of_arms,))
    self._counts = np.zeros((self._number_of_arms,))
