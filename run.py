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
    def __init__(self, opts, gpu_id=0):
        # Initialize Bolt class with options and GPU ID
        self.model_path = opts["model_path"]  # Path to the pre-trained model
        self.input_path = opts["input_path"]  # Path to the input video
        self.save_path = opts["save_path"]    # Path to save output frames
        self.interval = opts["interval"]      # Frame processing interval
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        # Create an instance of the BasicVSR model
        self.model = BasicVSR(num_feat=64, num_block=30)
        # Load the pre-trained model's weights
        self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
        # Set the model in evaluation mode and move it to the specified device
        self.model.eval()
        self.model.to(self.device)
        # Create the save path directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        # Extract frames from the input video and get its FPS
        self.img_list, self.fps = self.__vid2img(self.input_path)

    def __vid2img(self, input_path):
        # Function to convert a video into a list of frames
        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get frames per second (FPS) of the video
        success, image = vidcap.read()
        count = 0
        frame_list = []
        while success:
            frame_list.append(image)  # Append each frame to the list
            success, image = vidcap.read()
            count += 1
        return frame_list, fps

    def compare_images(self, target, ref):
        # Function to compare the quality of two images
        scores = []
        # Convert the images to PIL format
        PIL_image_gt = Image.fromarray(np.uint8(ref)).convert('RGB')
        PIL_image_output = Image.fromarray(np.uint8(target)).convert('RGB')
        # Calculate and append PSNR and SSIM scores
        scores.append(psnr(target, ref))
        scores.append(compare_ssim(PIL_image_output, PIL_image_gt))
        return scores

    def inference(self):
        # Perform super-resolution on frames and calculate quality metrics
        num_imgs = len(self.img_list)
        psnr = []
        ssim = []
        output_frame = []
        print("~~~~~Generating SR FRAMES~~~~~~")
        for idx in tqdm(range(0, num_imgs, self.interval)):
            interval = min(self.interval, num_imgs - idx)
            # Read a batch of frames for processing
            imgs = read_img_seq(self.img_list[idx:idx + interval], return_imgname=False)
            imgs_ = imgs.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Perform super-resolution using the model
                outputs = self.model(imgs_)
            outputs = outputs.squeeze()
            outputs = list(outputs)
            for num, img in enumerate(outputs):
                img_num = num + idx
                output = tensor2img(img)  # Convert the tensor to a numpy image
                output_frame.append(output)  # Store the super-resolved frame
                input_img = imgs[num]
                input_frame_numpy = self.img_list[img_num]
                input_frame_1024 = cv2.resize(input_frame_numpy, (1024, 1024))
                # Compare the super-resolved frame with the original and store quality metrics
                score = self.compare_images(output, input_frame_1024)
                psnr.append(score[0])
                ssim.append(score[1])
                if idx == 0 and num == 0:
                    h_stack = np.hstack((output, input_frame_1024))  # Create a horizontally stacked image
                # Save the super-resolved frame
                cv2.imwrite(os.path.join(self.save_path, f'{num}_{idx}_BasicVSR.png'), output)
        return np.mean(psnr), np.mean(ssim), h_stack, output_frame, self.fps
