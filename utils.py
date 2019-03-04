from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont

class crappification(object):
    """
    Applies the operations on image of the dataset
    Usage: parallel(crappification(path_lr,path_hr),ImageList.items) # next(iter(dataloader))[i]
    """
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

def save_preds(dl):
    i=0
    fnames = dl.dataset.items
    for j in dl:
        preds = unet.pred_batch(batch=j,reconstruct=True)
        for k in preds:
            k.save(path_gen/fnames[i].name)
            i+=1