# YoloMultispectralFusion-Coarse-to-fine-Registration
Created by Jiacheng Li, School of Automation Engineering, University of Electronic Science and Technology of China.  
Contact: jiacheng_li@std.uestc.edu.cn
## Run
### Weight Preparation
To prepare the model weights, you can download them from the official website: [Google Drive](https://drive.google.com/drive/folders/1Zxg4KORjy6h279bDqUCb7C3tw93Qxn-x?usp=drive_link) or [BaiduYun](https://pan.baidu.com/s/1iUBV7VL5fLOBWaV-MvRM5g?pwd=boqs)
### Environment
To set up the Python environment for this project, you can use the following command:

```bash
pip install -r requirements.txt
```
### Weight and Data Correspondence
To ensure proper detection, it's crucial to use the appropriate model weights that correspond to your data. 

- `ir_base.pt` and `vis_base.pt`: Corresponds to sample 2 and sample 4
- `ir_05.pt` and `vis_05.pt`: Corresponds to sample 5
- `ir_07.pt` and `vis_07.pt`: Corresponds to sample 7

Make sure to select the weight file that matches the specific sample you are working with and use it when running the detection process.
### Test
You can use the following command to run detection on example data:

```bash
python detect_multispectral.py --weights-vis vis-base.pt --weights-ir ir-base.pt --main-dir ./data/Demo_data_base --save-txt --save-conf
```
## YOLOv5

This project makes use of the open-source YOLOv5 object detection library, and we want to express our gratitude to the original authors of the YOLOv5 project and all the contributors who have contributed to it. Thanks to their hard work, we are able to leverage the powerful capabilities of YOLOv5 in this project.

**YOLOv5 Project Link:** [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

If you encounter any issues or need more information while using YOLOv5, please refer to the YOLOv5 project's documentation and community resources.
