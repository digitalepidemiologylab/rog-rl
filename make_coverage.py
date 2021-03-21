#!/usr/bin/env python

import os
import subprocess
import webbrowser
from urllib.request import pathname2url


def browser(pathname):
    webbrowser.open("file:" + pathname2url(os.path.abspath(pathname)))


subprocess.call(["xvfb-run", "coverage", "run", "--source", "rog_rl", "-m", "pytest"])
subprocess.call(["coverage", "report", "-m"])
subprocess.call(["coverage", "html"])

browser("coverage_html_report/index.html")
