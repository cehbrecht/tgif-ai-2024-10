# TGIF about AI

TGIF giving a short overview about artificial intelligence.

When: 2024-10-25
Where: DKRZ

## Google doc

https://docs.google.com/presentation/d/1rXnAXwFT3BRMr5CnHzMYfpprsIN6-pS3/edit#slide=id.g2bd8903ccd1_5_29


## Installation

This presentation uses [reveal-ck](http://jedcn.github.io/reveal-ck/).
See [documentation](http://jedcn.github.io/reveal-ck/installation/) for installation.

```
gem install reveal-ck
```

The slides `slides.md` are written in Markdown.

You can generate the presentation with reveal-ck:
```
$ reveal-ck generate
$ reveal-ck serve
http://localhost:10000
```

## Using RevealJS

See:
https://github.com/hakimel/reveal.js/blob/master/README.md

Export as PDF:
* https://revealjs.com/pdf-export/
* Open this URL and print: http://localhost:10000/?print-pdf


Overview mode with `ESC`:
https://github.com/hakimel/reveal.js/blob/master/README.md#overview-mode

## Install on macOS

See:
https://github.com/rbenv/rbenv#upgrading-with-homebrew

```
$ brew install rbenv ruby-build
$ rbenv init
# $ eval "$(rbenv init - zsh)"
$ rbenv install 2.7.0
$ rbenv global 2.7.0
$ gem install reveal-ck
```
