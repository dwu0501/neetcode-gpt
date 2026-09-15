import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = [np.exp(x - max(z)) for x in z]
        sumAll = np.sum(z)
        for i in range (len(z)):
            print(z[i])
            z[i] = z[i]/sumAll
        
        return np.round(z,4)
