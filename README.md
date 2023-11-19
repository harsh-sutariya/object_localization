# <font size=8> :label: Object Localisation Model </font>

This project aims to develop a series of open-source and strong fundamental image recognition models.
    |

## :toolbox: Checkpoints

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Backbone</th>
      <th>Data</th>
      <th>Illustration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>RAM++ (14M)</td>
      <td>Swin-Base</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Provide strong image tagging ability for any category.</td>
      <td><a href="https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth">Download  link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>RAM (14M)</td>
      <td>Swin-Large</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Provide strong image tagging ability for common category.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth">Download  link</a></td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tag2Text (14M)</td>
      <td>Swin-Base</td>
      <td>COCO, VG, SBU, CC3M, CC3M-val, CC12M</td>
      <td>Support comprehensive captioning and tagging.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
    </tr>
  </tbody>
</table>


## :running: Model Inference

### **Setting Up** ###

1. Install recognize-anything as a package:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

2. Or, for development, you may build from source

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```

3. ```bash
pip install -r requirements.txt
```

4. ```bash
pip install -e .
```

5. ```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```

6. ```bash
cd ./Grounded-Segment-Anything
```

7. ```bash
pip install -r ./requirements.txt
```

8. ```bash
pip install ./segment_anything
```

9. ```bash
pip install ./GroundingDINO
```

10. ```bash
cd ..
```

11. ```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```