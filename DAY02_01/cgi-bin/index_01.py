#!/usr/bin/env python

## 모듈 로딩 후 storage 인스턴스 생성
import cgi, codecs, sys

# WEB 인코딩 설정
sys.stdout=codecs.getwriter(encoding='utf-8')(sys.stdout.detach())

## 인자 처리
storage = cgi.FieldStorage()
filename = 'html/result.html'

def web_response(storage, filename):
    username = 'No DATA'
    password = 'No DATA'
    detailinfo = 'No Data'
    for name in storage:
        if name == 'username':
            username = storage.getvalue('username')
        elif name == 'password':
            password = storage.getvalue('password')
        elif name == 'detailinfo':
            detailinfo = storage.getvalue('detailinfo')

    with open(filename, 'r', encoding='utf-8') as f:
        
        print("Content-type: text/html\r\n\r\n")
        result = f.read()
        print(result.format(username, password, detailinfo))
        
    
web_response(storage, filename)


