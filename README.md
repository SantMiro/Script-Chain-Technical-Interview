# Script Chain Technical Interview

## Questions:

### 1. Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues? (paragraph format)

* First of all, since positional encoding is crucial for representing the order of the sequence, when stacking mulitple self-attention layers in the architecture, the model may not be able to scale well with longer sequences or with deeper layers.

* Secondly, without any regularization or normalization layers, overfitting is a risk as the depth of the architecture increseases. Deeper models have more parameters and can easily memorize the training data. Moreover, with multiple self-attention layers,  exploding gradients become a possible issue during training. Residual connections could be beneficial to ensure smoother gradient flow, but not a guarantee.

* Furthermore, computentional complexity becomes a problem, especially with long sequences. Self-attention has a $\mathcal{O}(n^2d)$ computational complexity. [[1]](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) Even though, self-attention has a shorter maximum path length than CNNs, "the quadratic computational complexity with respect to the sequence length makes self-attention prohibitively slow for very long sequences". [[1]](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)

* Finally, deeper architectures are harder to interpret and debug, so it is more difficult to understand the contribution of individual layers in the model and in place making it harder to identify possible improvments in some sections.

Reference:

    [1] Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press. [https://D2L.ai](https://D2L.ai)


### 2. Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)
