from fastai.conv_learner import *
from pathlib import Path
torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = True

PATH = Path('data/imagenet')
trn_pth = PATH/'train'

fnames_full,label_arr_full,all_labels = folder_source(PATH, 'train')
fnames_full = ['/'.join(Path(fn).parts[-2:]) for fn in fnames_full]
list(zip(fnames_full[:5],label_arr_full[:5]))

all_labels[:5]

np.random.seed(0)
keep_pct = 1
keeps = np.random.randn(len(fnames_full)) < keep_pct
fnames = np.array(fnames_full, copy=False)[keeps]
label_arr = np.array(label_arr_full,copy=False)[keeps]

arch = vgg16
sz_lr = 72         #72x72 low resolution input

scale,bs = 4,8   #scale-up factor
sz_hr = scale*sz_lr  # 144x144 output

class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0

aug_tfms = [RandomDihedral(tfm_y=TfmType.PIXEL)]

val_idxs = get_cv_idxs(len(fnames), val_pct=min(0.01/keep_pct,0.1))
((val_x,trn_x), (val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), np.array(fnames))
len(val_x), len(trn_x)

tfms = tfms_from_model(arch,sz_lr,tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms, sz_y = sz_hr)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=trn_pth)
md = ImageData(PATH,datasets,bs, num_workers=16, classes=None)

denorm = md.val_ds.denorm

def show_img(ims, idx, figsize=(5,5), normed=True, ax=None):
    if ax is None: 
        fig,ax = plt.subplots(figsize=figsize)
    if normed:
        ims = denorm(ims)
    else: 
        ims = np.rollaxis(to_np(ims),1,4)
    ax.imshow(np.clip(ims,0,1)[idx])
    ax.axis('off')

x, y = next(iter(md.val_dl))
x.size(),y.size()

idx = 2
fig,axes = plt.subplots(1,2,figsize=(9,5))
show_img(x,idx,ax=axes[0])
show_img(y,idx,ax=axes[1])

batches = [next(iter(md.aug_dl)) for i in range(9)]

fig,axes = plt.subplots(3,6,figsize=(18,9))
for i,(x,y) in enumerate(batches):
    show_img(x,idx,ax=axes.flat[i*2])
    show_img(x,idx,ax=axes.flat[i*2+1])

def conv(ni,nf,kernel_size=3,activation=False):
    layers = [nn.Conv2d(ni,nf,kernel_size,padding=kernel_size//2)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResSequential(nn.Module):
    def __init__(self,layers,res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)
    
    def forward(self,x):
        return x+self.m(x)*self.res_scale

def residual_block(nf):
    return ResSequential([conv(nf,nf,activation=True)],0.1)

def upsample(ni,nf,scale):
    layers = []
    for i in range(int(math.log(scale,2))):
        layers += [conv(ni,nf*4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)

class SrResNet(nn.Module):
    def __init__(self,nf,scale):
        super().__init__()
        features = [conv(3,64)]    # 3channels to 64 channels for richer spatial space
        for i in range(8):
            features.append(residual_block(64))  #stride 1 blocks so dimensions don't change  
        features += [conv(64,64), upsample(64,64,scale),
                        nn.BatchNorm2d(64),conv(64,3)]
        
        self.features = nn.Sequential(*features)    
        
    def forward(self,x):
        return self.features(x)

## Pixel Loss

m = to_gpu(SrResNet(64,scale))
m = nn.DataParallel(m)
learn = Learner(md,SingleModel(m),opt_fn=optim.Adam)
learn.crit = F.mse_loss

learn.lr_find(start_lr=1e-5,end_lr=10000)
learn.sched.plot()

learn.fit(lr,1,cycle_len=1,use_clr_beta=(40,10))

x,y = next(iter(md.val_dl))
preds = learn.model(VV(x))


idx = 7                           
show_img(y,idx,normed=False)     #input

show_img(preds,idx,normed=False)     #output

show_img(x,idx,normed=True)

def icnr(x, scale=2, init=nn.init.kaiming_normal):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel

m_vgg = vgg16(True)
blocks = []
for i,o in enumerate(children(m_vgg)):
    if isinstance(o,nn.MaxPool2d):
        blocks.append(i-1)
blocks, [m_vgg[i] for i in blocks]

vgg_layers = children(m_vgg)[:23]       #after 22, layers are course enough
m_vgg = nn.Sequential(*vgg_layers).cuda().eval()    #Vgg would not be trained
set_trainable(m_vgg,False)

def Flatten(x):
    return x.view(x.size(0),-1)

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

class FeatureLoss(nn.Module):              #Perceptual Loss
    def __init__(self,m,layer_ids,layer_wgts):
        super().__init__()
        self.m,self.wgts = m,layer_wgts
        self.sfs = [SaveFeatures(m[i]) for i in layer_ids]
        
    def forward(self,input, target, sum_layers=True):
        self.m(VV(target.data))             #target->image that we're trying to create
        res = [F.l1_loss(input,target/100)]  #Pixel Loss
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [F.l1_loss(Flatten(inp.features),Flatten(targ))*wgt
               for inp,targ,wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers:
               res = sum(res)
        return res
    
    def close(self):
        for o in self.sfs:
            o.remove()

m = SrResNet(64,scale)    #how much to scale up by

conv_shuffle = m.features[10][0][0]
kernel = icnr(conv_shuffle.weight,scale=scale)
conv_shuffle.weight.data.copy_(kernel);

m = to_gpu(m)

learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)

t = torch.load(learn.get_model_path('sr-samp0'), map_location=lambda storage,loc:storage)
learn.model.load_state_dict(t,strict=False)

learn.freeze_to(999)

for i in range(10,13):
    set_trainable(m.features[i],True)

conv_shuffle = m.features[10][2][0]
kernel = icnr(conv_shuffle.weight, scale=scale)
conv_shuffle.weight.data.copy_(kernel);

learn = Learner(md,SingleModel(m), opt_fn=optim.Adam)

learn.set_data(md)

learn.crit = FeatureLoss(m_vgg,blocks[:3],[0.2,0.7,0.1])

lr = 6e-3
wd = 1e-7

learn.save('sample0')  
#learn.load('sample0')

learn.lr_find(1e-4,0.1,wds=wd,linear =True)

learn.sched.plot(n_skip_end=1)

learn.fit(lr,1,cycle_len=2,wds=wd,use_clr=(20,10))

learn.save('sample1')       
#To load learn.load('sample1')

learn.unfreeze()

learn.fit(lr/3,1,cycle_len=1,wds=wd,use_clr=(20,10))

learn.save('a')

learn.sched.plot_loss()

def plot_ds_img(idx, ax=None, figsize=(7,7), normed=True):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    im = md.val_ds[idx][0]
    if normed: im = denorm(im)[0]
    else:      im = np.rollaxis(to_np(im),0,3)
    ax.imshow(im)
    ax.axis('off')

fig,axes=plt.subplots(6,6,figsize=(20,20))
for i,ax in enumerate(axes.flat): plot_ds_img(i+100,ax=ax, normed=True)

x,y = md.val_ds[9]

y=y[None]



learn.model.eval()
preds = learn.model(VV(x[None]))
x.shape,y.shape,preds.shape

learn.crit(preds,V(y),sum_layers=False)

learn.crit.close()

_,axes=plt.subplots(1,2,figsize=(14,7))
show_img(x[None], 0, ax=axes[0])
show_img(preds,0, normed=True, ax=axes[1])

