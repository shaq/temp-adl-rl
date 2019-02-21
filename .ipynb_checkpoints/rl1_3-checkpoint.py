
class UCB(object):

  def __init__(self, number_of_arms):
    self._number_of_arms = number_of_arms
    self.name = 'ucb'
    self.reset()

  def step(self, previous_action, reward):
    pass

  def reset(self):
    pass
