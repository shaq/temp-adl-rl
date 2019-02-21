
class UCB(object):

  def __init__(self, number_of_arms):
    self._number_of_arms = number_of_arms
    self.name = 'ucb'
    self.reset()

  def step(self, previous_action, reward):
    if previous_action is not None:
      r = reward
      
      # Updating values
      self.t += 1.0
      self._counts[previous_action] += 1.0
      alpha = 1 / self._counts[previous_action]
      self._estimates[previous_action] += alpha * (r - self._estimates[previous_action])
      
      # Getting next action with upper confidence bound
      upper_bound = np.sqrt(np.log(self.t) / (self._counts))
      return np.argmax(self._estimates + upper_bound)
      
    return np.random.randint(self._number_of_arms)

  def reset(self):
    """
    Resets all the statistics (estimated rewards and counts) to zero.
    
    This is also used to initialise these statistics on initialisation.
    """
    self.t = 0.
    self._estimates = np.zeros((self._number_of_arms,))
    self._counts = np.zeros((self._number_of_arms,))
