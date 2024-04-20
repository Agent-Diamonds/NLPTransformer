import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('../')


from Modules.modules import * 
from params import params

### Sanity check on posemb
# Create an instance of JointEmbedding
model = JointEmbedding(params['vocab_size'],
                       params['hidden_size'],
                       64,
                       params['dropout_prob'],
                       'cpu')

# Visualize the positional embeddings
positional_embeddings = model.position.squeeze(0).detach().cpu().numpy()
plt.figure(figsize=(12, 6))
plt.plot(positional_embeddings[:, 0], label='Position 0')
plt.plot(positional_embeddings[:, 1], label='Position 1')
plt.plot(positional_embeddings[:, 2], label='Position 2')
plt.title('Visualization of Positional Embeddings')
plt.xlabel('Position')
plt.ylabel('Value')
plt.legend()
plt.show()


