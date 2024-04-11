#!/usr/bin/env python

import cgi, codecs, sys
import cgitb

sys.stdout=codecs.getwriter(encoding='utf-8')(sys.stdout.detach())
cgitb.enable()

# 웹 페이지의 form 태그 내의 input 태그 입력값 가져오기
storage = cgi.FieldStorage()

print("Content-type: text/html\n")
html = '''
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head>
<title>test</title>
<h1>This is my cgi script</h1>
<img src='../example.raw' alt='이미지 업로드' width='300' height='300'/>
{}
'''

# 서버에 이미지 저장
with open('./example.raw', 'wb') as f:
    f.write(storage['이미지파일'].value)

print(html.format(storage.keys()))
