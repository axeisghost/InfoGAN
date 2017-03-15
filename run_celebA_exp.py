
# coding: utf-8

# # Reimplementing the CelebA experiment
# 
# In this notebook, I'm trying to reimplement the CelebA experiment results from the awesome [InfoGAN paper](https://arxiv.org/pdf/1606.03657v1.pdf) (Chen et al.).
# 
# My relevant additions to this repository (which contains the code published by the authors) are adding the "celebA" model to [infogan/models/regularized_gan.py](infogan/models/regularized_gan.py) and some small adjustments to [infogan/algos/infogan_trainer.py](infogan/algos/infogan_trainer.py) to allow for generating samples after training.

# In[14]:

# get_ipython().magic(u'pylab inline')


# In[15]:

from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
from infogan.misc.celebA_dataset import CelebADataset
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime
import os

import numpy as np

from matplotlib import pyplot as plt
from display_utils import display_images
import display_utils


# In[16]:

root_log_dir = "logs/celebA"
root_checkpoint_dir = "ckt/celebA"
batch_size = 128
updates_per_epoch = 5    # How often to run the logging.
checkpoint_snapshot_interval = 1000  # Save a snapshot of the model every __ updates.
max_epoch = 15


# In[17]:

# The "C.3 CelebA" input settings:
# "For this task, we use 10 ten-dimensional categorical code and 128 noise variables, resulting in a concatenated dimension of 228.."
c3_celebA_latent_spec = [
    (Uniform(128), False),  # Noise
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
    (Categorical(10), True),
]
c3_celebA_image_size = 32


# In[18]:

dataset = CelebADataset(202599)  # The full dataset is enormous (202,599 frames).

print("Loaded {} images into Dataset.".format(len(dataset.raw_images)))
print("Split {} images into training set.".format(len(dataset.train.images)))
print("Image shape: ",dataset.image_shape)


# In[6]:

# print("Displaying some training Images...\n Click to play!")
# display_images([frame.reshape(dataset.image_shape) for frame in dataset.train.images[:30]])


# In[10]:

model = RegularizedGAN(
    output_dist=MeanBernoulli(dataset.image_dim),
    latent_spec=c3_celebA_latent_spec,  # Trying with the above celebA latent_spec.
    batch_size=batch_size,
    image_shape=dataset.image_shape,
    # Trying with my new celebA network!
    network_type="celebA",
)


# In[11]:

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
exp_name = "celebA_model_celebA_codes_color_img-align-celeba_10_%s" % timestamp

log_dir = os.path.join(root_log_dir, exp_name)
checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

mkdir_p(log_dir)
mkdir_p(checkpoint_dir)

algo = InfoGANTrainer(
    model=model,
    dataset=dataset,
    batch_size=batch_size,
    exp_name=exp_name,
    log_dir=log_dir,
    checkpoint_dir=checkpoint_dir,
    max_epoch=max_epoch,
    updates_per_epoch=updates_per_epoch,
    snapshot_interval=checkpoint_snapshot_interval,
    info_reg_coeff=1.0,
    generator_learning_rate=1e-3,  # original paper's learning rate was 1e-3
    discriminator_learning_rate=2e-4,  # original paper's learning rate was 2e-4
)


# In[12]:

#algo.visualize_all_factors()  # ... what does this do?


# In[13]:

sess = tf.Session()

algo.train(sess=sess)


# In[ ]:




# In[ ]:




# ## Using the Trained Model: Generating Images
# 
# Alright! Now we've trained the model on our data, and we can use it to generate some new images!
# 
# We can just reuse the tiny piece of the TensorFlow graph that generates fake samples, $x$, from the learned distribution. We'll reuse the same `sess` variable that we used for training, so that all the variables still hold their learned values!

# In[10]:

def make_one_hot(length, value):
    v = np.zeros(length)
    v[value] = 1
    return v
def make_z(latent_spec, vals, noise = None):
    ''' noise - if specified will use provided noise, otherwise will generate noise from noise_dim. '''
    if noise is None:
        noise = np.random.rand(latent_spec[0][0].dim)
    
    codes = [make_one_hot(10, v) if isinstance(latent_spec[i+1][0],Categorical) else [v] for i,v in enumerate(vals)]
    return np.concatenate([noise]+codes)


# In[11]:

def generate_images_for_codes(latent_spec, codes, noise = None):
    ''' codes = 10 values 0-10 which represent the GAN codes (z). '''
    # Unfortunately, for now, I have to generate batch_size images at a time still.
    custom_z = np.asarray([make_z(latent_spec, codes, noise) for _ in range(batch_size)])
    return sess.run(algo.fake_x, feed_dict={algo.use_manual_z_input:1, algo.z_input: custom_z})


# In[13]:

# from ipywidgets import interact, interactive, fixed
# c=(0,10,1)
# @interact(z0=c,z1=c,z2=c,z3=c,z4=c,z5=c,z6=c,z7=c,z8=c,z9=c, num_images=(1,50,1), __manual=True)
# def images_from_codes(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9, num_images=10):
#     images = generate_images_for_codes(c3_celebA_latent_spec, [z0,z1,z2,z3,z4,z5,z6,z7,z8,z9][:len(c3_celebA_latent_spec)-1])
#     print("Displaying sampled images as movie. Click to play.")
#     return display_images([frame.reshape(dataset.image_shape) for frame in images[:num_images,:]])


# In[ ]:

# get_ipython().system(u'tensorboard --logdir /logs/celebA/celebA_model_celebA_codes_color_img-align-celeba_10_2017_03_15_02_20_02 ')


# In[ ]:




# In[ ]:




# In[ ]:



