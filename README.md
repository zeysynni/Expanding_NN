# Expanding Neural Networks for Solving Cart-Pole-Swing-Up

- Trains a policy that solve the non-linear control problem Cart-Pole-Swing-Up. 
- The policy takes five states as input and gives one action as output.
- The policy is trained with Policy Gradients, which employs Natural Gradients
- Ocasionally the product of the network Natural Gradient vector for a expanded architecture with itself under the Fisher-Norm will be computed.
- Network will be expanded when through training no progress are made and this product is larger than a threshold.
