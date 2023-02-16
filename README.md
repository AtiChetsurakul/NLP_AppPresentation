- ## Project Here
    - Resume Praser
    - Product Review analysis


## To use


#### you can find tutorial on how to run flask app on `flask Doc`.


#### I use chatGPT to generate a jsonl pipeline for education title extracting and partial copied from Tonson
<img src = 'how_my_web_perform/chatgpt.png'>


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
<img src = 'how_my_web_perform/uploadpage2.png'>

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
<img src = 'how_my_web_perform/hw4form.png'>
- ### and this is result
<<img src = 'how_my_web_perform/hw4result.png'>>


-----------------------



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
