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

  def get_baseline(self):
    if self._baseline:
      t = np.sum(self._counts)
      return self.R_t / t
    return self._baseline
    
  def step(self, previous_action, reward):
    T = self._temperature
    if previous_action is not None:
      r = reward
      
      # Updating action count, cummulative reward, and baseline
      self._counts[previous_action] += 1.0
      self.R_t += r
      b = self.get_baseline()
      
      # Updating action preferences
      ind_a = np.zeros((self._number_of_arms,))
      ind_a[previous_action] = 1.0
      phi = self._estimates / T
      self._estimates += ((self._lr * (r - b)) / T) * (ind_a - self._policy)

      # Policy iteration
      self._policy = np.exp(phi) / np.sum(np.exp(phi))
      probs = np.exp(self._policy) / np.sum(np.exp(self._policy))
      
      # New action-selection
      a = np.random.choice(probs, p=probs)
      action = np.argmax(probs == a)
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
