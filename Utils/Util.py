# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/6/1 19:37
# @Function:
import hashlib
handle = "https://tfhub.dev/google/universal-sentence-encoder/4"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())