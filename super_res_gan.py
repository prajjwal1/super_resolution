import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from utils import *

path = Path('/home/jupyter/data_proto/')
path_hr = path/'img_align_celeba'
path_lr = path/'image_gen'

from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont

il = ImageItemList.from_folder(path_hr)
parallel(crappifier(path_lr,path_hr),il.items)

bs,size = 32,128

arch = models.resnet34
src = ImageImageList.from_folder(path).random_split_by_pct(0.1, seed=42)

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

data_gen = get_data(bs,size)

wd=1e-3
y_range = (-3,3.)
loss_gen = MSELossFlat()

unet = unet_learner(data=data_gen,arch=arch,wd=wd,blur=True,norm_type=NormType.Weight,
                        self_attention=True,y_range=y_range,loss_func=loss_gen)

unet.fit_one_cycle(2)

unet.unfreeze()
unet.fit_one_cycle(3,slice(1e-6,1e-3))

unet.save('unet1')

dir_gen = 'image_gen'
path_gen = path/dir_gen
path_gen.mkdir(exist_ok=True)

def save_preds(dl):
    i=0
    fnames = dl.dataset.items
    for j in dl:
        preds = unet.pred_batch(batch=j,reconstruct=True)
        for k in preds:
            k.save(path_gen/fnames[i].name)
            i+=1


save_preds(data_gen.fix_dl)

classes = ['img_align_celeba','image_gen']

def crit_data(classes,bs,size):
    src = (ImageList.from_folder(path,include=classes)
                        .random_split_by_pct(0.1,seed=42)
                        .label_from_folder(classes=classes)
                        .transform(get_transforms(max_zoom=2.),size=size)
                        .databunch(bs=bs*2)
                        .normalize(imagenet_stats))
    src.c=3
    return src

data_crit = crit_data(classes,bs=bs,size=size)
data_crit.show_batch(rows=4,ds_type=DatasetType.Train,img_size=3)

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

critic = Learner(data_crit, gan_critic(),metrics=accuracy_thresh_expand,loss_func=loss_critic,wd=wd)
critic.load('critic2');

critic.fit_one_cycle(6,1e-3)


unet = unet_learner(data=data_gen,arch=arch,wd=wd,blur=True,norm_type=NormType.Weight,self_attention=True,y_range=y_range,loss_func=loss_gen)
unet.load('unet1');

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(unet, critic, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

learn.save('start1')


learn.fit(10,1e-4)


learn.save('gan1')


learn.data = get_data(32,192)

learn.fit(10,(1e-4)/2)




