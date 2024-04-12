#!/usr/bin/env python

## 모듈 로딩 후 storage 인스턴스 생성
import cgi, cgitb, codecs, sys, io
import datetime
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from konlpy.tag import Mecab
import logging 
import pickle
from PIL import Image
cgitb.enable()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WEB 인코딩 설정
sys.stdout=codecs.getwriter(encoding='utf-8')(sys.stdout.detach())

## 인자 처리
storage = cgi.FieldStorage()
filename = 'html/result.html'

class TextMulti(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_Class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_Class)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, _hidden = self.rnn(embedded)
        return self.fc(output)

numCLASS = 9
class VGG11_pretrained(nn.Module):
    def __init__(self):
        super(VGG11_pretrained, self).__init__()
        self.model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, numCLASS)

        for name, model in self.model.named_parameters():
            if name.find('classifier') < 0:
                model.requires_grad = False
            else:
                model.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x

class VGG11_pretrained(nn.Module):
    def __init__(self):
        super(VGG11_pretrained, self).__init__()
        self.model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, numCLASS)

        for name, model in self.model.named_parameters():
            if name.find('classifier') < 0:
                model.requires_grad = False
            else:
                model.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x
    
preprocess_vgg11 = transforms.Compose(
    [
    # transforms.Resize((224 // 2, 224 //2)),
    transforms.ToTensor(),
    transforms.Normalize(
     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
]
)



def load_vocab(file_path):
    with open(file_path, 'rb') as f:  
        vocab = pickle.load(f)
    return vocab

dialect ={
    0 : '강원도 사투리',
    1 : '경상도 사투리',
    2 : '전라도 사투리',
    3 : '제주도 사투리',
    4 : '충청도 사투리',
    5 : '표준말'
}

imageclass = ['abstract_painting',
 'cityscape',
 'genre_painting',
 'illustration',
 'landscape',
 'portrait',
 'religious_painting',
 'sketch_and_study',
 'still_life'
]
imageclass_beautify = [
    '추상화 그림',
    '도시 풍경',
    '장르화 그림',
    '일러스트',
    '풍경화',
    '인물화',
    '종교적 그림',
    '스캐치한 그림',
    '정적삶 그림'
]


vocab = load_vocab('./model/vocab_multi.pkl')
model = torch.load('./model/model_multi.pt', map_location=DEVICE)
model_img = torch.load('./model/model_image_genre.pth', map_location=DEVICE)
mecab = Mecab()

def predict(model, text):
    text_pipeline = lambda x: vocab(x)
    with torch.no_grad():
        text = torch.tensor(text_pipeline(mecab.morphs(text)), dtype=torch.int64).to(DEVICE)       
        offsets = torch.tensor([0]).to(DEVICE)
        pre = model(text, offsets)
        pre = torch.softmax(pre, dim=1)
    return pre

def web_response(storage, filename):
    detailinfo = ''
    finalresult = '입력된 데이터가 없습니다.'
    file = None
    finalimgresult = '업로드된 사진이 없습니다.'
    rawname = './image/result_{}.raw'.format(datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    # with open(rawname, 'wb') as f: 
    #     Image.new(mode="RGBA", size=(200, 200), color='white').save(f)
    

    if 'detailinfo' in storage.keys():
        detailinfo = storage.getvalue('detailinfo')
        if detailinfo:
            
            result = predict(model, detailinfo)
            resultloc = torch.argmax(result, dim=1)
            resultloc = resultloc.item()
            finalresult = dialect[resultloc]
        
    if 'fileupload' in storage.keys():
        
        file = storage['fileupload'].value
        with open(rawname, 'wb') as f:
            f.write(file)
        
            imageio = io.BytesIO(file)
            img = Image.open(imageio).resize((64,64)).convert("RGB")
    
            imageori = img.copy()
            imageT = preprocess_vgg11(imageori).to(DEVICE)
            imageT = imageT.unsqueeze(0)
            finalimgresult = imageclass_beautify[torch.argmax(model_img(imageT), dim=1).item()]
            
    with open(filename, 'r', encoding='utf-8') as f:
        print("Content-type: text/html\r\n\r\n")
        result = f.read()
        print(result.format(rawname, detailinfo, finalresult, finalimgresult))



  
web_response(storage, filename)


