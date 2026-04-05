import ctypes
from ctypes import *
import time
import cv2
import numpy as np
import win32gui
from pathlib import Path
import cv2
import pydirectinput as ms
import win32api
import win32con
from trtyolo import TRTYOLO
import supervision as sv
import csv



#记录pid曲线图
data_file = open('pid_data.csv', 'w', newline='')
csv_writer = csv.writer(data_file)
csv_writer.writerow(['time', 'err_x', 'err_y', 'move_x', 'move_y'])
start_time = time.time()



#准星位置
win_target=[240,267]#窗口坐标的位置

MOVE_THRESHOLD = 10 # 死区阈值（像素），小于此值不移动


# 加载之前构建的TensorRT引擎
model = TRTYOLO("best_static.engine", task="detect", swap_rb=True, profile=False)#加载训练模型（改为trt加速）
#dxgi初始化
dxgi = ctypes.CDLL('E:/YOLO11/ultralytics-8.3.39/cs2_test/test/dxgi4py-master/dxgi4py.dll')
dxgi.grab.argtypes = (POINTER(ctypes.c_ubyte), ctypes.c_int, c_int, c_int, c_int)
dxgi.grab.restype = POINTER(c_ubyte)


# 创建标注器（可选，如果需要画框的话）
box_annotator = sv.BoxAnnotator()


#_______找到目标窗口并进行初始化________
windowTitle = 'Counter-Strike 2'
hwnd = win32gui.FindWindow(None, windowTitle)

if hwnd == 0:
    print("找不到目标窗口")
    exit()

windll.user32.SetProcessDPIAware()

dxgi.init_dxgi(hwnd)

#________获取窗口的坐标和尺寸_________
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
shotLeft, shotTop = 0, 0
height = bottom - top
width = right - left
shot = np.ndarray((height, width, 4), dtype=np.uint8)
shotPointer = shot.ctypes.data_as(POINTER(c_ubyte))



#定义要监控的特定区域
# 格式: (左上角x偏移, 左上角y偏移, 宽度, 高度)
monitor_region = {
    'x': 721,      # 距离窗口左边721像素
    'y': 318,      # 距离窗口上边318像素
    'width': 500,  # 区域宽度500像素
    'height': 500 # 区域高度500像素          以射击准星为中心作一个500*500的自瞄方形框
}



#_______________PID算法区____________
Kp=2.0
Ki=0
Kd=0.5
summ_x=0
summ_y=0
old_err_x=0
old_err_y=0
#______________终止隔开符_____________________


#_______鼠标移动函数_________
def move(x, y):
    """移动鼠标"""
    # 1. 移动鼠标到坐标 (x, y)相对坐标，x正值向→，y正值向↓
    #ms.moveRel(x, y, relative=True)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
    #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, x, y, 0, 0)

#_______终止隔开符__________



# 帧率计算
fps_counter = 0
fps = 0
start_time = time.time()
last_fps_time = start_time
last_frame_time = start_time


#说明
print("CS2 实时监控 - 按 'q' 退出")

#主程序
try:
    while True:
        #帧率计算
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time

        # 瞬时FPS
        if frame_time > 0:
            instantaneous_fps = 1.0 / frame_time
        else:
            instantaneous_fps = 0


        # 捕获一帧
        buffer = dxgi.grab(
            shotPointer,
            monitor_region['x'],  # 左上角x偏移
            monitor_region['y'],  # 左上角y偏移
            monitor_region['width'],   # 区域宽度
            monitor_region['height']   # 区域高度
            )
        Target_x=win_target[0]+monitor_region['x'] #窗口准星坐标转化为屏幕坐标
        Target_y=win_target[1]+monitor_region['y']
        if buffer:
            image = np.ctypeslib.as_array(buffer, shape=(monitor_region['height'], monitor_region['width'], 4))
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            #用模型对该图片进行推理
            detections = model.predict(img)
            if len(detections) > 0:
                mask = (detections.confidence >= 0.5) & (detections.class_id == 2)
                if mask.any():
                    filtered = detections[mask]  # 直接索引得到新的Detections对象
                    centers_screen = (filtered.xyxy[:, :2] + filtered.xyxy[:, 2:]) / 2  # 计算中心点 (x, y)
                    centers_screen[:, 0] += monitor_region['x']
                    centers_screen[:, 1] += monitor_region['y']

                    #print("[{} {}]".format(Target_x,Target_y))
                    # 找出离准星最近的目标索引
                    target_pos = np.array([Target_x, Target_y])
                    distances = np.linalg.norm(centers_screen - target_pos, axis=1)  # 或直接用平方距离
                    #print(distances)
                    best_idx = np.argmin(distances)
                    x1, y1, x2, y2 = filtered.xyxy[best_idx].astype(int)  # 区域图像内的坐标
                    conf = filtered.confidence[best_idx]
                    cls = filtered.class_id[best_idx]


                    #_______以上为选中最近的那个目标并给出中心坐标，下方针对此坐标进行坐标运算并进行移动________


                    # 目标中心（区域图像坐标）
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center = [center_x, center_y]

                    # 准星在区域图像内的坐标（用于画箭头）
                    win_local_x = win_target[0] - monitor_region['x']
                    win_local_y = win_target[1] - monitor_region['y']
                    win_local = [win_local_x, win_local_y]

                    # 画箭头：从准星指向目标中心
                    cv2.arrowedLine(img, win_target, center, (255, 0, 0), thickness=1, tipLength=0.1)

                    # 目标中心在屏幕上的绝对坐标
                    screen_centerx = center_x + monitor_region['x']
                    screen_centery = center_y + monitor_region['y']

                    # 计算鼠标移动量（将准星移至目标）
                    delta_x = Target_x - screen_centerx
                    delta_y = Target_y - screen_centery
                    distance = (delta_x ** 2 + delta_y ** 2) ** 0.5  # 欧几里得距离

                    if distance >= MOVE_THRESHOLD: #加入死区，在死区范围内不移动：防止平凡跳动

                        err_x=delta_x
                        err_y=delta_y
                        summ_x+=err_x
                        summ_y+=err_y

                        move_x = int(Kp*err_x+Kd*(err_x-old_err_x)+Ki*(summ_x))  # 注意方向：我们要将准星移向目标，所以移动 -delta_x
                        move_y = int(Kp*err_y+Kd*(err_y-old_err_y)+Ki*(summ_y))

                        old_err_x = err_x
                        old_err_y = err_y

                        #print(move_x)
                        #print(move_y)
                        move(-move_x, -move_y)
                        current_time = time.time() - start_time
                        csv_writer.writerow([current_time, err_x, err_y, -move_x, -move_y])
                        print(err_x)
                    else:
                        # 距离小于阈值，可以完全不移动，或者选择直接精确移动到位（但可能会引起抖动）
                        # 这里选择不移动
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        old_err_x=0
                        old_err_y=0 #目标解决后旧误差清零，对新目标重新计算
                        pass




            else:
                filtered = None





            # 计算并显示帧率
            fps_counter += 1
            if current_time - last_fps_time >= 1.0:
                fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
                #print(f"平均FPS: {fps:.1f}, 瞬时FPS: {instantaneous_fps:.1f}")

            # 在图像上显示FPS信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Inst: {instantaneous_fps:.1f}", (10, 70), font, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Size: {width}x{height}", (10, 110), font, 0.7, (255, 255, 255), 2)

            # 显示图像
            #cv2.imshow('CS2_go', img)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    # 清理资源
    data_file.close()
    cv2.destroyAllWindows()
    dxgi.destroy()
    print("\n监控结束")
