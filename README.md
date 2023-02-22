- ## Project Here
    - Resume Praser
    - Product Review analysis


## To use


<!-- #### you can find tutorial on how to run flask app on `flask Doc`.


#### I use chatGPT to generate a jsonl pipeline for education title extracting and partial copied from Tonson
<img src = 'how_my_web_perform/chatgpt.png'>
 -->

----------------

- First, `pip install -r requirement.txt`
- then on bash -> $ `export FLASK_APP=main`
- then $`flask run`
-------------------
## how it work
- ### Homepage
 <img src = 'how_my_web_perform/homepage.png'>

-------------------------
- ## Project1 Resume Stealer
- ### UploadPage
<!-- <img src = 'how_my_web_perform/uploadpage2.png'> -->

- ### File uploading
<img src = 'how_my_web_perform/uploadafile.png'>

- ### Result
<img src = 'how_my_web_perform/result.png'>

---------------
- ## Project2 Tweet to review product
<!-- - ### I have 2 excute
<img src = 'how_my_web_perform/excute0.png'>

- ### second one
<img src = 'how_my_web_perform/excute1.png'>

- ## Maybe we should sent email to `Elon Musk` to sell `twitter` out or just let it went bankrupt. Since he do something in TWITTER API and that will not be free anymore T-T . -->

<!-- - ## Anyway, we still have our freind REDDIT -->
### So our site now is like

<img src = 'how_my_web_perform/hwtwmp.png'>

- ### Our form for this hw
<!-- <img src = 'how_my_web_perform/hw4form.png'> -->
- ### and this is result
<img src = 'how_my_web_perform/hw4result.png'>


-----------------------

- ## Project 3 
- Our site are now add new feature

<img src= 'how_my_web_perform/dick0.png'>

- How it work?

<img src= 'how_my_web_perform/dick1.png'>
<img src= 'how_my_web_perform/dick2.png'>



- An Editable n stackable result are show as this  

<img src= 'how_my_web_perform/dick3.png'>
<img src= 'how_my_web_perform/dick4.png'>
<img src= 'how_my_web_perform/dick5.png'>


------------------------------



## code Struceture for my flask web app
```bash
.
├── data_stealer.py                         #PROJECT 1
├── form_.py
├── instance
│   ├── create_dir.py
│   └── requirements.txt
├── main.py
├── pySCRPT                                 # PROJECT 2
│   ├── modude_.py
│   ├── pycache_loader_.py
│   ├── runner.py
│   └── test_praw.py
├── static
│   ├── assets
│   │   ├── css
│   │   │   ├── fontawesome-all.min.css
│   │   │   ├── main.css
│   │   │   └── noscript.css
│   │   ├── js
│   │   │   ├── breakpoints.min.js
...
│   │   │   ├── main.js
│   │   │   └── util.js
│   │   ├── sass
│   │   │   ├── base
│   │   │   │   ├── _page.scss
│   │   │   │   ├── _reset.scss
│   │   │   │   └── _typography.scss
│   │   │   ├── components
│   │   │   │   ├── _actions.scss
│   │   │   │   ├── _banner.scss
...
│   │   │   │   └── _wrapper.scss
│   │   │   ├── layout
│   │   │   │   └── _wrapper.scss
│   │   │   ├── libs
│   │   │   │   ├── _breakpoints.scss
│   │   │   │   ├── _functions.scss
│   │   │   │   ├── _html-grid.scss
│   │   │   │   ├── _mixins.scss
│   │   │   │   ├── _vars.scss
│   │   │   │   └── _vendor.scss
│   │   │   ├── main.scss
│   │   │   └── noscript.scss
│   │   └── webfonts
│   │       ├── fa-brands-400.eot
...
│   │       ├── fa-solid-900.woff
│   │       └── fa-solid-900.woff2
│   ├── edu_skill.jsonl
│   ├── files
│   │   └── joke.csv
│   ├── images
│   │   ├── banner.jpg
...
│   │   ├── gallery
│   │   │   ├── fulls
│   │   │   │   ├── 01.jpg
│   │   │   │   ├── 02.jpg
...
│   │   │       └── 12.jpg
│   │   ├── pic01.jpg
│   │   ├── pic02.jpg
...
│   │   └── spotlight03.jpg
│   └── skills.jsonl
└── templates
    ├── footer.html
    ├── header.html
    ├── index.html
    ├── index_.html
    ├── stealed.html
    ├── stealed2.html
    └── upload.html

```


-----------------------
## System

```bash
root@0746646b7288:~# nvidia-smi
Thu Feb 16 06:27:08 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:84:00.0 Off |                  N/A |
| 44%   71C    P2   185W / 250W |   2388MiB / 11264MiB |     95%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  On   | 00000000:85:00.0 Off |                  N/A |
| 37%   63C    P2   199W / 250W |   5385MiB / 11264MiB |     56%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce ...  On   | 00000000:88:00.0 Off |                  N/A |
| 22%   28C    P8     6W / 250W |      3MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce ...  On   | 00000000:89:00.0 Off |                  N/A |
| 22%   27C    P8     5W / 250W |      3MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.108.03   Driver Version: 510.108.03   CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   46C    P5    15W /  N/A |     82MiB /  6144MiB |     34%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+



```

