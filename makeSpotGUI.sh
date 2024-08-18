#!/bin/sh
rm -rf dist/*; python -m build; python -m pip install dist/spotgui*.tar.gz -U
python -m mkdocs build
