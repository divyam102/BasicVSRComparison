from run import Bolt  # Import the Bolt class from the 'run' module
import cv2  # Import OpenCV for video processing
import numpy as np  # Import NumPy for numerical operations

if __name__ == '__main__':
    # Define options for the first Bolt object
    opts_1 = {
        "model_path":"/content/drive/MyDrive/chkpts/net_g_300000.pth",
        "input_path":"/content/BasicVSRComparison/video1_256p.mp4",
        "save_path":"/content/BasicVSRComparison/video_frame_path_3",
        "interval":7,
    }

    # Define options for the second Bolt object
    opts_2 = {
        "model_path":"/content/drive/MyDrive/chkpts/net_g_400000.pth",
        "input_path":"/content/BasicVSRComparison/video1_256p.mp4",
        "save_path":"/content/BasicVSRComparison/video_frame_path_4",
        "interval":7,
    }

    # Create two Bolt objects with different options
    obj_3k = Bolt(opts_1)  # Instantiate the first Bolt object
    obj_4k = Bolt(opts_2)  # Instantiate the second Bolt object

    # Perform super-resolution and quality assessment for both objects
    psnr_3k, ssim_3k, h_stack_3k, output_frame_3k, fps_3k = obj_3k.inference()
    psnr_4k, ssim_4k, h_stack_4k, output_frame_4k, fps_4k = obj_4k.inference()

    # Save horizontally stacked images to see the difference between the original frame and SR frame
    cv2.imwrite(f"/content/BasicVSRComparison/stack_3k/img_stack.jpg", h_stack_3k)
    cv2.imwrite(f"/content/BasicVSRComparison/stack_4k/img_stack.jpg", h_stack_4k)

    # Define output video paths
    pathOut_3k = "/content/BasicVSRComparison/video_3k/output.avi"
    pathOut_4k = "/content/BasicVSRComparison/video_4k/output.avi"

    # Create video writers for the output videos
    out_3k = cv2.VideoWriter(pathOut_3k, cv2.VideoWriter_fourcc(*'DIVX'), fps_3k, (1024, 1024))
    out_4k = cv2.VideoWriter(pathOut_4k, cv2.VideoWriter_fourcc(*'DIVX'), fps_4k, (1024, 1024))

    # Write super-resolved frames to the output videos
    for i in range(len(output_frame_3k)):
        out_3k.write(output_frame_3k[i])
        out_4k.write(output_frame_4k[i])

    # Release the video writers
    out_3k.release()
    out_4k.release()

    # Print PSNR and SSIM values for both checkpoints
    print(f"PSNR and SSIM value of 3K checkpoint are {psnr_3k} and {ssim_3k}")
    print(f"PSNR and SSIM value of 4K checkpoint are {psnr_4k} and {ssim_4k}")
