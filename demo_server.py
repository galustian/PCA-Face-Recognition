from flask import Flask, request, render_template
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main_get():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def main_post():
    file1 = request.files['your_face']
    file2 = request.files['other_face']
    
    im1 = Image.open(file1).convert('L')
    im2 = Image.open(file2).convert('L')

    # new_size = (im2.size[0] * 0.2, im2.size[1] * 0.2)
    # img_size was set to (118, 179)
    # to be correct: (179.20000000000002, 118.4)
    im1.thumbnail((179, 118))
    im2.thumbnail((179, 118))

    # im1.show()
    # im1.save("./test.jpg")
    im1_arr_2D = np.array(im1)
    im2_arr_2D = np.array(im2)
    
    im1_arr = np.ravel(im1_arr_2D).reshape(-1, 1)
    im2_arr = np.ravel(im2_arr_2D).reshape(-1, 1)

    im1_missing_pixels = [177 for i in range(U.shape[0] - im1_arr.shape[0])]
    im2_missing_pixels = [177 for i in range(U.shape[0] - im2_arr.shape[0])]
    im1_arr = np.append(im1_arr, im1_missing_pixels)
    im2_arr = np.append(im2_arr, im2_missing_pixels)
    
    im1_y = U.T @ im1_arr
    im2_y = U.T @ im2_arr

    im_diff = im1_y - im2_y

    euclid_dist = np.linalg.norm(im_diff)

    return render_template('index.html', score=int(euclid_dist))

if __name__ == '__main__':
    U = np.load('../principal-components.npy')

    app.run(debug=True)