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


@app.route('/evalg')
def evalg():
    return render_template('heavyidx.html')


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
