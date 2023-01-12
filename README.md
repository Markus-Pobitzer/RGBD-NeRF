# RGB-D NeRF: Depth supervised NeRF on synthetic depth maps

[**Project**](https://github.com/Markus-Pobitzer/RGBD-NeRF) | [**Report**](./Resources/RGBD_NeRF.pdf)

This is the pyTorch code for the course project in [263-0600-00L  Research in Computer Science](https://www.vorlesungen.ethz.ch//Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=162285&semkez=2022W&ansicht=LEHRVERANSTALTUNGEN&lang=en)
conducted at ETH Zürich and supervised by Dr. [Sergey Prokudin](https://scholar.google.de/citations?user=xSywCzAAAAAJ&hl=en). It extends  NeRF can be extended NeRF with synthetic depth information to reduce the needed number of input images.


 [Markus Pobitzer](https://markus-pobitzer.github.io/markuspobitzer/)

---

Virtual reality (VR) and augmented reality (AR) immerse
the user in a new digital world. However, representing realworld scenes and objects digitally is very challenging. Realistic lighting and high details are hard to model. An approach that solves some of the mentioned shortcomings was
introduced with Representing Scenes as Neural Radiance
Fields for View Synthesis (NeRF). NeRF can produce photorealistic novel views but needs many RGB input images to
train. In this work, we explore how NeRF can be extended
with synthetic depth information to reduce the needed number of input images.

## Results

NeRF trained with 2 views:
<p align="center">
  <img src="./Resources/rgb-fern.gif"  width="800" />
</p>

RGB-D NeRF trained with 2 views:
<p align="center">
  <img src="./Resources/rgb-d-fern.gif"  width="800" />
</p>

---


## Quick Start

### Dependencies

Install requirements:
```
pip install -r requirements.txt
```

### Data

Download the Fern dataset [here](https://drive.google.com/drive/folders/1L4itFnmYqbaeoJCCs2ClLVPXjZY_iY7L?usp=sharing).
And then put it into the `./data` folder.

### Pre-trained Models and Results

The pretrained models and evaluations on the test set can be found here: [link](https://drive.google.com/drive/folders/1bPA-tvvl7XZGOGlos0MJ35dniR9Hat9y?usp=sharing)

For example, our pre-trained model for Fern trained on 2 views can be found under:
```
├── results.zip
│   ├── Fern
│   ├── ├── DS-RGB-D-Fern-2
│   ├── ├── ├── 050000.tar
```
And the output of NeRF:
```
├── results.zip
│   ├── Fern
│   ├── ├── DS-RGB-Fern-2
│   ├── ├── ├── 050000.tar
```

#### Training

To train a DS-NeRF on the example `fern` dataset:
```
python run_nerf.py --config configs/fern_d.txt
```

It will create an experiment directory in `./logs`, and store the checkpoints and rendering examples there.

There is also a config file for the scene ship: `ship_d.txt`. Make sure you downloaded the dataset first.

---

## Acknowledgments

This code borrows heavily from [DS-NeRF](https://github.com/dunbar12138/DSNeRF) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). 
Special thanks go out to the supervisor of this work, Dr.
Sergey Prokudin for proposing this interesting topic and for
the kind guidance throughout the work.
