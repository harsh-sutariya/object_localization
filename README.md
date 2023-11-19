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
      <th>Checkpoint</th>
      <th>Path to be stored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Grounded SAM</td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">Download  link</a></td>
      <td>./Grounded-Segment-Anything/</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RAM (14M)</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth">Download  link</a></td>
      <td>./pretrained/</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tag2Text (14M)</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
      <td>./pretrained/</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAM</td>
      <td><a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">Download  link</a></td>
      <td>./Grounded-Segment-Anything/</td>
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

3. 
```bash
pip install -r requirements.txt
```

4. 
```bash
pip install -e .
```

5. delete the grounded-segment-anything folder first.
```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```

6. 
```bash
cd ./Grounded-Segment-Anything
```

7. 
```bash
pip install -r ./requirements.txt
```

8. 
```bash
pip install ./segment_anything
```

9. 
```bash
pip install ./GroundingDINO
```

10. 
```bash
cd ..
```

11. 
```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```