- ## Project Here

  - Resume Praser
  - Product Review analysis
  - Auto coding in python
  - Translator

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

<img src = 'how_my_web_perform/hw4result.png'>

-----------------------

- ## Project 3

- Our site are now add new feature

<img src= 'how_my_web_perform/dick0.png'>

- How it work?

<img src= 'how_my_web_perform/dick1.png'>
<img src= 'how_my_web_perform/dick2.png'>

- with this doc

``` html
<form action="/evalg" method="post">
    {% if generate %}
    <textarea type="text" name='old' style="border:none; outline:none;" id="input-field2"
        oninput="resizeTextarea()">{{generate}}</textarea>
    {% endif %}
    <input type="text" name='dick' style="border:none; outline:none;" id="input-field" />
    <button type="submit">Submit</button>
</form>
<script>

    function generateRandomString(length) {
        let result = '';
        const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        const charactersLength = characters.length;
        for (let i = 0; i < length; i++) {
            result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
        return result;
    }
    const inputField = document.getElementById("input-field");
    inputField.addEventListener("keyup", function (event) {
        if (event.keyCode === 13) {
            const inputValue = inputField.value;
            const randomString = generateRandomString(5); // Change the length as per your requirement
            inputField.value = `${inputValue} ${randomString}`;
        }
    });

    function resizeTextarea() {
        const textarea = document.getElementById("input-field2");
        textarea.style.height = "1px";
        textarea.style.height = (25 + textarea.scrollHeight) + "px";
    }

</script>
<style>
    #input-field2 {
        resize: none;
        height: 500px;
        overflow-y: scroll;
    }
</style>
```

- An Editable n stackable result are show as this  

<img src= 'how_my_web_perform/dick3.png'>
<img src= 'how_my_web_perform/dick4.png'>
<img src= 'how_my_web_perform/dick5.png'>

------------------------------

## Translator Project

- TH -> ENG translator

<img src= 'how_my_web_perform/MT1.png'>
<img src= 'how_my_web_perform/MT2.png'>


-------------------------------
- CODE Auto Complete with `Transformer and Beamsearh`
<img src= 'how_my_web_perform/decep0.png'>
<img src= 'how_my_web_perform/decep1.png'>
<img src= 'how_my_web_perform/decep2.png'>