from flask import Flask, render_template, redirect, url_for, flash, request, send_from_directory
from flask_bootstrap import Bootstrap
from flask_ckeditor import CKEditor
from flask_login import UserMixin, login_user, LoginManager, login_required, current_user, logout_user
from form_ import *
# from flask_sqlalchemy import SQLAlchemy
from flask_wtf.file import FileField
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
# from data_stealer import *
import os
from functools import wraps
import pyperclip3 as pyclip
import datetime as dt
import praw
from transformers import pipeline
from transformers import Conversation

PASSWORD_STR = os.environ.get('adminpassw', 'pw')

WTF_CSRF_SECRET_KEY = PASSWORD_STR


dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, root_path=dir_path)

app.config["DEBUG"] = True
bootstrap = Bootstrap(app)
app.config['sheet'] = 'static/files'
app.config['sheetpath'] = os.environ.get('SHEETPATH_CV', 'joke.csv')

app.config.update(dict(
    SECRET_KEY=PASSWORD_STR,
    WTF_CSRF_SECRET_KEY=WTF_CSRF_SECRET_KEY
))


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/evalm', methods=['GET', 'POST'])
def evalm():

    from ati_trans import inferencecer_MT
    from ati_trans import tranF_module
    if request.method == 'POST':
        old = request.form.get('old')
        dick = request.form.get('dick')
        # print(dick)
        generated = inferencecer_MT.translation(
            dick, 'general', '/Users/atichetsurakul/Desktop/JAN23/nlp123clone/NLP_labsession/Hw6_MLtranslate/models/Seq2SeqPackedAttention_general.pt', inferencecer_MT.device)
        if old is not None:
            return render_template('heavyidxMT.html', generate=old+'\n'+dick + '='+' '.join(generated))
        else:
            return render_template('heavyidxMT.html', generate=dick + '='+' '.join(generated))
    else:
        # return render_template('form.html')
        return render_template('heavyidxMT.html')


@app.route('/evalcommu', methods=['GET', 'POST'])
def evalcommu():
    generator = pipeline(
        "conversational", model="srv/DialoGPT-medium-Breaking_Bad", max_length=30, pad_token_id=0)
    # import dicKKUtill
    if request.method == 'POST':
        old = request.form.get('old')
        dick = request.form.get('dick')
        print(dick)

        # Conversation("Say my name!")
        # generator(Conversation(dick))
        # generated = dicKKUtill.generate(dick, 30, .8, dicKKUtill.model, dicKKUtill.tokenizer,
        #                                 dicKKUtill.vocab, dicKKUtill.device, 3407)
        if old is not None:
            return render_template('heavyidxcommu.html', generate=old+'\n'+generator(Conversation(dick)))
        else:
            return render_template('heavyidxcommu.html', generate=generator(Conversation(dick)))
    else:
        # return render_template('form.html')
        return render_template('heavyidxcommu.html')


@app.route('/evalg', methods=['GET', 'POST'])
def evalg():

    import dicKKUtill
    if request.method == 'POST':
        old = request.form.get('old')
        dick = request.form.get('dick')
        print(dick)
        generated = dicKKUtill.generate(dick, 30, .8, dicKKUtill.model, dicKKUtill.tokenizer,
                                        dicKKUtill.vocab, dicKKUtill.device, 3407)
        if old is not None:
            return render_template('heavyidx.html', generate=old+'\n'+' '.join(generated))
        else:
            return render_template('heavyidx.html', generate=' '.join(generated))
    else:
        # return render_template('form.html')
        return render_template('heavyidx.html')


@app.route('/evalDecepticon', methods=['GET', 'POST'])
def evalDecepticon():

    import transerDick as dicKKUtill
    if request.method == 'POST':
        old = request.form.get('old')
        dick = request.form.get('dick')
        print(dick)
        generated = dicKKUtill.generate(dick, 30, .8, dicKKUtill.model, dicKKUtill.tokenizer,
                                        dicKKUtill.vocab, dicKKUtill.device, 3407)
        if old is not None:
            return render_template('heavyidxdecep.html', generate=old+'\n'+' '.join(generated))
        else:
            return render_template('heavyidxdecep.html', generate=' '.join(generated))
    else:
        # return render_template('form.html')
        return render_template('heavyidxdecep.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # form = PdfForm(meta={'csrf': False})
    form = PdfForm()
    if form.validate_on_submit():
        f = form.pdff_.data
        page = int(form.page.data)
        filename = secure_filename(f.filename)
        # print(f)
        f.save(os.path.join(
            app.instance_path, filename
        ))
        return redirect(url_for('result_table', file_name=filename, page=page))

    return render_template('upload.html', form=form, head='HomeWork NO. 3 Resume stealer')


@app.route('/checker', methods=['GET', 'POST'])
def checker():
    form = checkerForm()
    if form.validate_on_submit():
        f = form.textS.data
        n = int(form.numm.data)
        from pySCRPT import secKey
        from pySCRPT import pycache_loader_
        client, secret, _ = secKey.ret_id()
        reddit = praw.Reddit(
            client_id=client, client_secret=secret, user_agent='ati')
        subreddit = reddit.subreddit(f)
        topics = [*subreddit.top(limit=n)]
        fifty_sen = [n.title for n in topics]
        test = pycache_loader_.inference_classification(fifty_sen, True)
        # test2 = str(sum(test)/len(test))
        score = list([(s, int(i)) for s, i in test])
        return render_template('stealed2.html', score=score)

    return render_template('upload.html', form=form, head='Hw4 : Dunno the topic')


@app.route('/stealing/<file_name>/<int:page>')
def result_table(file_name, page):
    path = os.path.join(app.instance_path, file_name)
    import data_stealer

    skill, edu = data_stealer.to_read(path, page=page)
    os.remove(path)
    return render_template('stealed.html', ret=' '.join(skill), eddu=' '.join(edu))
    pass


@app.route('/download')
def download_csv():
    return send_from_directory(app.config['sheet'], path=app.config['sheetpath'], as_attachment=True)
