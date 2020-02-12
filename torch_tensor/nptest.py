import numpy as np

X = np.array([
              [ 1,2,3,4,
                5,6,7,8
               ],
                [ 1,2,3,4,
                5,6,7,8
               ],
              ])

print(X.shape)

print(X.resize(3,9,9))
print(X)
