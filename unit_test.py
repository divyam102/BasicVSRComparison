from run import Bolt
import cv2
import numpy as np 

if __name__ == '__main__':
    opts_1 = {
        "model_path":"/home/ubuntu/projects/python/divyam/BasicVSR_PlusPlus/chkpts/net_g_300000.pth",
        "input_path":"/home/ubuntu/projects/python/divyam/BasicVSR/video1_256p.mp4",
        "save_path":"/home/ubuntu/projects/python/divyam/BasicVSR/video_frame_path_3",
        "interval":7,
    }
    opts_2 = {
        "model_path":"/home/ubuntu/projects/python/divyam/BasicVSR_PlusPlus/chkpts/net_g_400000.pth",
        "input_path":"/home/ubuntu/projects/python/divyam/BasicVSR/video1_256p.mp4",
        "save_path":"/home/ubuntu/projects/python/divyam/BasicVSR/video_frame_path_4",
        "interval":7,
    }
    obj_3k = Bolt(opts_1)
    obj_4k = Bolt(opts_2)
    psnr_3k,ssim_3k, h_stack_3k, output_frame_3k, fps_3k = obj_3k.inference()
    psnr_4k,ssim_4k, h_stack_4k, output_frame_4k, fps_4k = obj_4k.inference()
    cv2.imwrite(f"/home/ubuntu/projects/python/divyam/BasicVSR/stack_3k/img_stack.jpg", h_stack_3k)
    cv2.imwrite(f"/home/ubuntu/projects/python/divyam/BasicVSR/stack_4k/img_stack.jpg", h_stack_4k)
    pathOut_3k = "/home/ubuntu/projects/python/divyam/BasicVSR/video_3k/output.avi"
    pathOut_4k = "/home/ubuntu/projects/python/divyam/BasicVSR/video_4k/output.avi"
    out_3k = cv2.VideoWriter(pathOut_3k,cv2.VideoWriter_fourcc(*'DIVX'), fps_3k, (1024, 1024))
    out_4k = cv2.VideoWriter(pathOut_4k,cv2.VideoWriter_fourcc(*'DIVX'), fps_4k, (1024, 1024))
    for i in range(len(output_frame_3k)):
        # writing to a image array
        out_3k.write(output_frame_3k[i])
        out_4k.write(output_frame_4k[i])
    out_3k.release()
    out_4k.release()

    print(f"PSNR and ssim value of 3K chckpoint are {psnr_3k} and {ssim_3k}")
    print(f"PSNR and ssim value of 4K chckpoint are {psnr_4k} and {ssim_4k}")
    # print(np.mean(psnr), np.mean(ssim), fps)