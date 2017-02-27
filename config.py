# -*- coding:utf-8 -*-


import logging
import os

FORMS_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'SEC-Edgar-Data'))

FAIL_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fail_company_list.txt'))

HTTP_TIMEOUT = 30