import cv2
import torch
import numpy as np
from torch.nn import functional as F
import time
import argparse

def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model('train_log', -1)
        print("Loaded v2.x HD model.")
    except:
        try:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model('train_log', -1)
            print("Loaded v3.x HD model.")
        except:
            from model.RIFE import Model
            model = Model()
            model.load_model('train_log', -1)
            print("Loaded ArXiv-RIFE model")
    
    model.eval()
    model.device()
    return model, device

def main():
    """实时补帧主函数
    从摄像头读取视频流并进行实时补帧，支持自定义分辨率。
    """
    # 初始化模型
    model, device = init_model()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 推荐 CAP_DSHOW，Linux 可以直接 VideoCapture(0)

    # 设置摄像头分辨率
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1920, help='摄像头宽度分辨率')
    parser.add_argument('--height', type=int, default=1080, help='摄像头高度分辨率')
    args = parser.parse_args()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 获取摄像头分辨率
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 调整尺寸为32的倍数
    w = (w // 32) * 32
    h = (h // 32) * 32
    print(f"实际处理分辨率: {w}x{h}（已调整为32的倍数）")
    
    # 初始化帧率计算变量
    prev_time = time.time()
    fps_list = []
    
    # 读取第一帧
    ret, lastframe = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        return
    
    while True:
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换帧格式并上传到GPU
        I0 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        
        # 调整输入尺寸为32的倍数
        I0 = I0[:, :, :h, :w]
        I1 = I1[:, :, :h, :w]
        
        # 生成中间帧
        middle = model.inference(I0, I1, 1.0)  # scale=1.0
        middle = (middle[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
        middle = np.ascontiguousarray(middle, dtype=np.uint8)
        
        # 计算帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        fps_list.append(fps)
        if len(fps_list) > 30:  # 保持最近30帧的平均
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        prev_time = current_time
        
        # 在画面上显示帧率
        cv2.putText(middle, f'FPS: {avg_fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示原始帧和插值帧
        # cv2.imshow('Original', frame)
        cv2.imshow('Interpolated', middle)
        
        # 更新lastframe
        lastframe = frame
        
        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()