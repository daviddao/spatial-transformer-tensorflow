# Spatial Transformer Network

Tensorflow Implementation of Spatial Transformer Networks as described in [1] and based on [2].

<div align="center">
  <img src="http://i.imgur.com/gfqLV3f.png"><br><br>
</div>

### API 
    
Implements a spatial transformer layer as described in [1].
Based on [2] and edited for Tensorflow.

#### How to use
```
transformer(U, theta, downsample_factor=1)
```
    
#### Parameters

    U : float 
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels]. 
    theta: float   
        The output of the
        localisation network should be [num_batch, 6].
    downsample_factor : float
        A value of 1 will keep the original size of the image
        Values larger than 1 will downsample the image. 
        Values below 1 will upsample the image
        example image: height = 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
        
    
#### Notes
To initialize the network to the identity transform init ``theta`` to :

```python
identity = np.array([[1., 0., 0.],
                    [0., 1., 0.]]) 
identity = identity.flatten()
theta = tf.Variable(initial_value=identity)
```        

### References

[1] Jaderberg, Max, et al. "Spatial Transformer Networks." arXiv preprint arXiv:1506.02025 (2015)

[2] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
