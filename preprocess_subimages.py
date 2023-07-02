import os
from tqdm import tqdm
cnt = 0
for path in tqdm(os.listdir('./DIV2K_train_LR_bicubic/X2_sub/')):
    a = os.path.join('./DIV2K_train_LR_bicubic/X2_sub', path)
    img = cv2.imread(a)
    b = os.path.join('./DIV2K_train_HR_sub', path)
    img_ = cv2.imread(b)
    img = cv2.resize(img, dsize=None, fx=2, fy=2)
    img = tfs.ToTensor()(img).numpy()
    img_ = tfs.ToTensor()(img_).numpy()
    img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    img_ = cv2.cvtColor(img_.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    
    psnr = peak_signal_noise_ratio(img[...,0], img_[..., 0])

    if psnr > 35:
        os.remove(a)
        os.remove(b)
        cnt += 1
print(cnt)