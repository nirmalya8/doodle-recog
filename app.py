from flask import Flask, render_template, jsonify, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit',methods=['POST'])#,'GET'])
def file_uploaded():
    file = request.files['myfile']
    file.save(file.filename)
    return jsonify({'file':file.filename})

if __name__ == '__main__':
    app.run(debug=True)