from flask import Flask, render_template, request
import torch
from test_model import predict
from Neural_Architecture import LSTM_architecture
from math import ceil

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

vocab_size = 194345
output_size = 1
embedding_dim = 200
hidden_dim = 128
number_of_layers = 2
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
model.load_state_dict(torch.load("model", map_location=torch.device('cpu')))
seq_length = 30

@app.route("/", methods=['GET', 'POST'])
def hello():
    flag = False
    type_of_tonal = ""
    prob = 0
    name = ""
    if request.method == 'POST':
        flag = True
        if request.form["submit_button"]:
            name1 = request.form['text_tonal']
            if len(name1) != 0:
                name = name1
            type_of_tonal, pos_prob = predict(model, name, seq_length)
            if type_of_tonal == "Негативное сообщение":
                prob = ceil((1 - pos_prob)*100)
            else:
                prob = ceil(pos_prob*100)

    return render_template('main.html', flag = flag, type_of_tonal = type_of_tonal, percent = "{} %".format(prob), text = name)

if __name__ == "__main__":
    app.run()


