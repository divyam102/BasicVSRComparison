import cv2
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm 
import torch
from SSIM_PIL import compare_ssim
from evaluation_metric import psnr
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


class Bolt:
    def __init__(self, opts, gpu_id = 0):
        self.model_path = opts["model_path"]
        self.input_path = opts["input_path"]
        self.save_path = opts["save_path"]
        self.interval = opts["interval"]
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = BasicVSR(num_feat=64, num_block=30)
        self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
        self.model.eval()
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)
        self.img_list, self.fps = self.__vid2img(self.input_path)
    def __vid2img(self, input_path):
        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image = vidcap.read()
        count = 0
        frame_list = []
        while success:
            frame_list.append(image)
            success,image = vidcap.read()

            count += 1
        return frame_list, fps
    def compare_images(self, target, ref):
        scores = []
        PIL_image_gt = Image.fromarray(np.uint8(ref)).convert('RGB')
        PIL_image_output = Image.fromarray(np.uint8(target)).convert('RGB')
        scores.append(psnr(target, ref))
        scores.append(compare_ssim(PIL_image_output, PIL_image_gt))
        return scores
    def inference(self):
        num_imgs = len(self.img_list)
        psnr = []
        ssim = []
        output_frame = []
        print("~~~~~Generating SR FRAMES~~~~~~")
        for idx in tqdm(range(0, num_imgs, self.interval)):
            interval = min(self.interval, num_imgs - idx)
            imgs = read_img_seq(self.img_list[idx:idx + interval], return_imgname=False)
            imgs_ = imgs.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs_)
            outputs = outputs.squeeze()
            outputs = list(outputs)
            for num, img in enumerate(outputs):
                img_num = num + idx
                output = tensor2img(img)
                output_frame.append(output)
                input_img = imgs[num]
                input_frame_numpy = self.img_list[img_num]
                input_frame_1024 = cv2.resize(input_frame_numpy, (1024, 1024))
                score = self.compare_images(output, input_frame_1024)
                psnr.append(score[0])
                ssim.append(score[1])
                if idx == 0 and num == 0:
                    h_stack = np.hstack((output, input_frame_1024))
                cv2.imwrite(os.path.join(self.save_path, f'{num}_{idx}_BasicVSR.png'), output)
        return psnr,ssim, h_stack, output_frame, self.fps

