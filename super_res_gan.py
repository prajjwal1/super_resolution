#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *


# In[2]:


path = Path('/home/jupyter/data_proto/')
path_hr = path/'img_align_celeba'
path_lr = path/'image_gen'


# In[3]:


from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont

class crappifier(object):
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr              
        
    def __call__(self, fn, i):       
        dest = self.path_lr/fn.relative_to(self.path_hr)    
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 96, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w,h = img.size
        q = random.randint(10,70)
        ImageDraw.Draw(img).text((random.randint(0,w//2),random.randint(0,h//2)), str(q), fill=(255,255,255))
        img.save(dest, quality=q)


# In[32]:


il = ImageItemList.from_folder(path_hr)
parallel(crappifier(path_lr,path_hr),il.items)


# In[4]:


bs,size = 32,128


# In[5]:


arch = models.resnet34
src = ImageImageList.from_folder(path).random_split_by_pct(0.1, seed=42)


# In[6]:


def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


# In[7]:


data_gen = get_data(bs,size)
#data_gen.show_batch(4)


# In[8]:


wd=1e-3
y_range = (-3,3.)
loss_gen = MSELossFlat()


# In[13]:


#data_gen.show_batch()


# In[52]:


unet = unet_learner(data=data_gen,arch=arch,wd=wd,blur=True,norm_type=NormType.Weight,
                        self_attention=True,y_range=y_range,loss_func=loss_gen)


# In[53]:


unet.fit_one_cycle(2)


# In[55]:


unet.unfreeze()
unet.fit_one_cycle(3,slice(1e-6,1e-3))


# In[56]:


unet.show_results(4)


# In[57]:


unet.save('unet1')


# In[58]:


dir_gen = 'image_gen'
path_gen = path/dir_gen
path_gen.mkdir(exist_ok=True)


# In[67]:


def save_preds(dl):
    i=0
    fnames = dl.dataset.items
    for j in dl:
        preds = unet.pred_batch(batch=j,reconstruct=True)
        for k in preds:
            k.save(path_gen/fnames[i].name)
            i+=1


# In[68]:


save_preds(data_gen.fix_dl)


# In[14]:


classes = ['img_align_celeba','image_gen']


# # Critic

# In[15]:


def crit_data(classes,bs,size):
    src = (ImageList.from_folder(path,include=classes)
                        .random_split_by_pct(0.1,seed=42)
                        .label_from_folder(classes=classes)
                        .transform(get_transforms(max_zoom=2.),size=size)
                        .databunch(bs=bs*2)
                        .normalize(imagenet_stats))
    src.c=3
    return src


# In[16]:


data_crit = crit_data(classes,bs=bs,size=size)
data_crit.show_batch(rows=4,ds_type=DatasetType.Train,img_size=3)


# In[17]:


loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


# In[18]:


critic = Learner(data_crit, gan_critic(),metrics=accuracy_thresh_expand,loss_func=loss_critic,wd=wd)


# In[19]:


critic.load('critic2');


# In[26]:


critic.fit_one_cycle(6,1e-3)


# In[28]:


#critic.save('critic2')


# In[ ]:


#unet,critic=None,None


# In[20]:


unet = unet_learner(data=data_gen,arch=arch,wd=wd,blur=True,norm_type=NormType.Weight,self_attention=True,y_range=y_range,loss_func=loss_gen)
unet.load('unet1');


# In[21]:


switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(unet, critic, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))


# In[28]:


learn.save('start1')


# In[ ]:


learn.fit(10,1e-4)


# In[ ]:


learn.save('gan1')


# In[ ]:


learn.data = get_data(32,192)


# In[ ]:


learn.fit(10,(1e-4)/2)


# In[ ]:


learn.show_results(rows=16)


# In[ ]:




