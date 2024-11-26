ERUS-10K Dataset Info.

File structure, for example:
--image (image frames caputred from videos)
--mask (binary mask for each image)
--video_label.csv (The video number and corresponding video label. video_id is the video number, video_label is the video label)
--bounding_box.json (The bounding box annotation for each frame extracted from the videos)

Note:
1. Only the original images are provided, you can use mode conversion code in the repository to generate augmentated datasets
3. video_label: 0-CRC T0 Stage, 1-CRC T1 Stage, 2-CRC T2 Stage, 3-CRC T3 Stage, 4-CRC T4 Stage in video_label.csv.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ERUS-10K 数据集信息

文件结构，例如：
--image（从视频中截取的图像帧）
--mask（每个图像的二值掩码）
--video_label.csv（视频编号和对应的视频标签。video_id 是视频编号，video_label 是视频标签）
--bounding_box.json（从视频中提取的每个帧的边界框注释）

注意：
1. 仅提供原始图像，您可以使用[ASTR github repository](https://github.com/yuncheng97/ASTR)中的模式转换代码来生成增强数据集
3. video_label：video_label.csv 中的 0-CRC T0 阶段、1-CRC T1 阶段、2-CRC T2 阶段、3-CRC T3 阶段、4-CRC T4 阶段。