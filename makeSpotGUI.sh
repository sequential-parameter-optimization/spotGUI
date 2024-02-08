#!/bin/sh
rm -f dist/spotGUI*; python -m build; python -m pip install dist/spotGUI*.tar.gz
python -m mkdocs build
