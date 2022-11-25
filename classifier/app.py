import torch
from model import LeNet 

classes = ['square', 'axe', 't-shirt', 'soccer_ball', 'book', 'sun', 'apple', 'calculator', 'face', 'pizza']
model = LeNet()
model.load_state_dict(torch.load("./Models/lenet2.pt"))
from model import LeNet 
def predict(img):
    try:
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
        # print(x.shape)
        with torch.no_grad():
            out = model(x)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        values, indices = torch.topk(probabilities, 5)
        confidences = {classes[i]: v.item() for i, v in zip(indices, values)}
        return confidences
    except TypeError:
        pass
    

import gradio as gr

gr.Interface(fn=predict, 
             inputs="sketchpad",
             outputs="label",
             live=True).launch()
