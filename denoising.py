import torch
import model as Model
import argparse
import os
import numpy as np
import yaml
import json
from collections import OrderedDict
import imageio.v2 as imageio
import cv2
import metrics

def read_a_img(path): # function of read one image file from $path$ --> np.ndarray of range [0.0, 1.0]
    return imageio.imread(path) / 65535. 

def save_a_img(img, path): # function of save $img$ (np.ndarray of range [0.0, 1.0]) to $path$
    return imageio.imwrite(path, (img*65535).astype(np.uint16))

if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Dn_Liver_128.yaml',
                        help='yaml file for configuration') ###

    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)

    json_str = ''
    with open(cfg['model']['cfg_path'], 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    model_cfg = json.loads(json_str, object_pairs_hook=OrderedDict)
    if len(model_cfg['gpu_ids']) > 1:
        model_cfg['distributed'] = True
    else:
        model_cfg['distributed'] = False

    model_cfg['path']['resume_state'] = cfg['model']['pretrained_path']
    root = cfg['data']['root']
    input_folder = cfg['data']['input_folder']
    cond_folder = cfg['data']['cond_folder'] if cfg['data']['cond_folder'] != 'None' else None
    output_folder = cfg['data']['output_folder']
    target_folder = cfg['data']['target_folder'] if cfg['data']['target_folder'] != 'None' else None
    res = cfg['data']['res']
    length = cfg['data']['len']
    if_ddim = cfg['diffusion']['ddim']
    ddim_eta = cfg['diffusion']['ddim_eta']
    mode = cfg['dn']['mode']
    lam0 = cfg['dn']['lam0']
    a, b, c = cfg['dn']['a'], cfg['dn']['b'], cfg['dn']['c']
    resume = cfg['dn']['resume']
    mean_num = cfg['dn']['mean_num']
    bs = cfg['dn']['bs']
    
    # data
    low_imgs = sorted(os.listdir(os.path.join(root, input_folder)))
    cond_imgs = sorted(os.listdir(os.path.join(root, cond_folder))) if cond_folder is not None else None
    target_imgs = sorted(os.listdir(os.path.join(root, target_folder))) if target_folder is not None else None
    if length == -1:
        length = len(low_imgs)
    os.makedirs(os.path.join(root, output_folder), exist_ok=True)
    inputs, conds, targets = [], [], []
    for i, low_name in enumerate(low_imgs):
        
        input = read_a_img(os.path.join(root, input_folder, low_name))
        input = cv2.resize(input, (res, res), cv2.INTER_CUBIC)

        if cond_folder is not None:
            cond = read_a_img(os.path.join(root, cond_folder, cond_imgs[i]))
            cond = cv2.resize(cond, (res, res), cv2.INTER_CUBIC)
        else:
            cond = None

        if target_folder is not None:
            target = read_a_img(os.path.join(root, target_folder, target_imgs[i]))
            target = cv2.resize(target, (res, res), cv2.INTER_CUBIC)
        else:
            target = None

        input = torch.Tensor(input).unsqueeze(0).cuda() * 2 - 1
        cond = torch.Tensor(cond).unsqueeze(0).cuda() * 2 - 1 if cond_folder is not None else None
        target = torch.Tensor(target).unsqueeze(0).cuda() * 2 - 1 if target_folder is not None else None
        inputs.append(input)
        conds.append(cond)
        targets.append(target)

    inputs = torch.stack(inputs, dim=0)
    conds = torch.stack(conds, dim=0) if cond_folder is not None else None
    targets = torch.stack(targets, dim=0) if target_folder is not None else None

    # model
    diffusion = Model.create_model(model_cfg)
    timesteps = list(range(0, 500, 20)) + list(range(500, 2000, 500)) + [1999]

    # denoising
    for i in range((length-1)//bs+1):

        input = inputs[i*bs:i*bs+bs if i*bs+bs < length else length, ...]
        n = input.shape[0]
        cond = conds[i*bs:i*bs+bs if i*bs+bs < length else length, ...] if cond_folder is not None else None
        input = torch.cat([input]*mean_num)
        cond = torch.cat([cond]*mean_num) if cond_folder is not None else None

        diffusion.inversion(input, cond, timesteps, ddim_eta, 
                            batch_size=n*mean_num, ddim=if_ddim, lambda1=torch.full(input.shape, lam0).to(input.device), a=a, b=b, c=c, resume=resume, mode=mode, continous=False)
   
        visuals = diffusion.get_current_visuals(sample=True)

        for j in range(n):
            denoised = torch.mean(visuals['SAM'][-(n*mean_num-j)::n, ...], dim=0)
            denoised_npy = denoised.clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5
            
            if target_folder is not None:
                input_npy = inputs[i*bs+j, ...].clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5
                target = targets[i*bs+j, ...].clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5
                psnr_org = metrics.calculate_psnr(input_npy*255., target*255.)
                ssim_org = metrics.calculate_ssim(input_npy*255., target*255.)
                psnr = metrics.calculate_psnr(denoised_npy*255., target*255.)
                ssim = metrics.calculate_ssim(denoised_npy*255., target*255.)
                print('%s-resolution %d, psnr %.2f to %.2f; ssim %.3f to %.3f'%(low_imgs[i*bs+j], res, psnr_org, psnr, ssim_org, ssim))
        
            save_a_img(denoised_npy, os.path.join(root, output_folder, low_imgs[i*bs+j]))
