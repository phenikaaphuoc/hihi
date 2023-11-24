import os
import skimage
import torch
import importlib
import zipfile
from models import dior_model  # Assuming dior_model is the name of the module (not the class)
import sys
import importlib
importlib.reload(dior_model)
from models.dior_model import DIORModel 
import os, json
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import utils.pose_utils as pose_utils

dataroot = 'data'
exp_name = 'DIORv1_64' 
epoch = 'latest'
netG = 'diorv1' 
ngf = 64
def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    if pose != None:
        print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])

        print(out.shape)
        return out
        if pose != None:
            out = np.concatenate((kpt, out),1)
    else:
        out = kpt
    return out

class Opt:
    def __init__(self):
        pass
if True:
    opt = Opt()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 8; opt.n_kpts = 18; opt.style_nc = 64
    opt.n_style_blocks = 4; opt.netG = netG; opt.netE = 'adgan'
    opt.ngf = ngf
    opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'; opt.init_gain = 0.02; opt.gpu_ids = None;
    opt.frozen_flownet = True; opt.random_rate = 1; opt.perturb = False; opt.warmup=False
    opt.name = exp_name
    opt.vgg_path = ''; opt.flownet_path = ''
    opt.checkpoints_dir = 'checkpoints'
    opt.frozen_enc = True
    opt.load_iter = 0
    opt.epoch = epoch
    opt.verbose = False



model = DIORModel(opt)
model.setup(opt)
# load data
from datasets.deepfashion_datasets import DFVisualDataset
Dataset = DFVisualDataset
ds = Dataset(dataroot=dataroot, dim=(256,176), n_human_part=8)


inputs = dict()
for attr in ds.attr_keys:
    inputs[attr] = ds.get_attr_visual_input(attr)
    
# define some tool functions for I/O
def load_img(pid, ds):
    if len(pid[0]) < 10: # load pre-selected models
        person = inputs[pid[0]]
        person = (i for i in person)
        pimg, parse, to_pose = person
        pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
    else: # load model from scratch
        person = ds.get_inputs_by_key(pid[0])
        person = (i for i in person)
        pimg, parse, to_pose = person
    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    if pose != None:
        import utils.pose_utils as pose_utils
        print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])

        if pose != None:
            out = np.concatenate((kpt, out),1)
    else:
        out = kpt
    return out

# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2], perturb=False):
    PID = [0,4,6,7]
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    if perturb:
        pimg = perturb_images(pimg[None])[0]
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
   
    
    # swap base garment if any
    gimgs = []
    for gid in gids:
        _,_,k = gid
        gimg, gparse, pose =  load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2])]

    # encode garment (overlay)
    garments = []
    over_gsegs = []
    oimgs = []
    for gid in ogids:
        oimg, oparse, pose = load_img(gid, ds)
        oimgs += [oimg * (oparse == gid[2])]
        seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
        over_gsegs += [seg]
    
    gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)
    
    return pimg, gimgs, oimgs, gen_img[0], to_pose
     
# download from https://www.stickpng.com/img/memes/doge/doge-facing-right
import cv2
import numpy as np
# download_from_gdrive("data","doge.png","1ZjEZVWGyLX7Mefrkc03uaEsspP0Nk_EL",iszip=False)

fn = "doge.png"
pid = ('plain', 3, None)

image = cv2.imread(fn, cv2.IMREAD_UNCHANGED) #Read RGBA image
# put the print on a blank canvas
x,y,h,w = 90,60,80,70
image = cv2.resize(image, (w,h))
bg = np.zeros((256,176,4))
bg[x:x+h,y:y+w] = image
image = bg

# crop the print image
trans_mask = image[:,:,3] != 0
image = image[:,:,2::-1].transpose(2,0,1)
image = (image / 255.0) * 2 - 1
image = image * trans_mask[None]


# run DiOr
pimg, parse, to_pose =  load_img(pid, ds) 
psegs = model.encode_attr(pimg[None], parse[None], to_pose[None], to_pose[None], [0,4,6,7])
gsegs = model.encode_attr(pimg[None], parse[None], to_pose[None], to_pose[None], [5,1,2])
# insert the print
print_image = torch.from_numpy(image).float()
print_fmap = model.netE_attr(print_image[None], model.netVGG)
print_mask = model.netE_attr.segmentor(print_fmap)

gsegs = gsegs[:1] + [(print_fmap, torch.sigmoid(print_mask))] + gsegs[1:] 


# generate
gen_img = model.netG(to_pose[None], psegs, gsegs)

# construct a copy-and-paste image for comparison
paste_img = image + pimg.cpu().detach().numpy() * (1 - trans_mask[None])

paste_img = torch.from_numpy(paste_img).float()


# display
output = torch.cat([gen_img[0]],2)


output = output.float().cpu().detach().numpy()


output = (output + 1) / 2
output = np.transpose((output * 255.0).astype(np.uint8), [1,2,0])

cv2.imshow("hi",cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
cv2.waitKey(5000)
cv2.destroyAllWindows()


pid = ("print", 0, None) # load the 0-th person from "print" group, NONE (no) garment is interested
# pose id (take this person's pose)
pose_id = ("print", 2, None) # load the 2-nd person from "print" group, NONE (no) garment is interested
# generate
pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, pose_id=pose_id)

output = plot_img(pimg,gimgs,oimgs,gen_img,pose)

cv2.imshow("hi",cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)