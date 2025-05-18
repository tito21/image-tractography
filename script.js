const imgElement = document.getElementById('imageSrc');
const inputElement = document.getElementById('fileInput');
const outputElement = document.getElementById('imageOutput');
const numLinesElement = document.getElementById('numLines');
const lengthLineElement = document.getElementById('lengthLine');
const lineWidthElement = document.getElementById('lineWidth');
const gradientCutoffElement = document.getElementById('gradientCutoff');
const startButtonElement = document.getElementById('startButton');

let offscreenCanvas = null;
let bar = null;
inputElement.addEventListener('change', (e) => {
    imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

startButtonElement.addEventListener('click', (e) => {
    if (imgElement.src) {
        imgElement.dispatchEvent(new Event('load'));
    }
}, false);


let src_gray, grad, vec_2, data;


function get_structural_tensor(img) {
    let grad_x = new cv.Mat();
    let grad_y = new cv.Mat();
    cv.Sobel(img, grad_x, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
    cv.Sobel(img, grad_y, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
    let grad = new cv.Mat();
    cv.magnitude(grad_x, grad_y, grad);

    let grad_xx = new cv.Mat();
    let grad_yy = new cv.Mat();
    let grad_xy = new cv.Mat();
    cv.multiply(grad_x, grad_x, grad_xx);
    cv.multiply(grad_y, grad_y, grad_yy);
    cv.multiply(grad_x, grad_y, grad_xy);
    let grad_xx_blur = new cv.Mat();
    let grad_yy_blur = new cv.Mat();
    let grad_xy_blur = new cv.Mat();
    cv.GaussianBlur(grad_xx, grad_xx_blur, { width: 3, height: 3 }, 1, 1, cv.BORDER_DEFAULT);
    cv.GaussianBlur(grad_yy, grad_yy_blur, { width: 3, height: 3 }, 1, 1, cv.BORDER_DEFAULT);
    cv.GaussianBlur(grad_xy, grad_xy_blur, { width: 3, height: 3 }, 1, 1, cv.BORDER_DEFAULT);
    return [grad_xx_blur, grad_yy_blur, grad_xy_blur, grad];
}


function get_eigenvalues(grad_xx, grad_yy, grad_xy) {
    // let lambda_1 = new cv.Mat(cv.CV_32F);
    // let lambda_2 = new cv.Mat(cv.CV_32F);
    let lambda_1 = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);
    let lambda_2 = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);

    // lambda_1 = 0.5 * (grad_xx + grad_yy + sqrt((grad_xx - grad_yy)^2 + 4 * grad_xy^2))
    // lambda_2 = 0.5 * (grad_xx + grad_yy - sqrt((grad_xx - grad_yy)^2 + 4 * grad_xy^2))

    cv.subtract(grad_xx, grad_yy, lambda_1);
    cv.multiply(lambda_1, lambda_1, lambda_1); // (grad_xx - grad_yy)^2
    let four_grad_xy = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F, new cv.Scalar(4));
    cv.multiply(grad_xy, four_grad_xy, four_grad_xy); // error
    cv.multiply(four_grad_xy, grad_xy, four_grad_xy); // 4 * grad_xy^2
    cv.add(lambda_1, four_grad_xy, lambda_1); // (grad_xx - grad_yy)^2 + 4 * grad_xy^2
    let sqrt = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);
    cv.sqrt(lambda_1, sqrt); // sqrt((grad_xx - grad_yy)^2 + 4 * grad_xy^2)
    let trace = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);
    cv.add(grad_xx, grad_yy, trace); // grad_xx + grad_yy
    cv.add(trace, sqrt, lambda_1);
    cv.subtract(trace, sqrt, lambda_2);
    let half = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F, new cv.Scalar(0.5));
    cv.multiply(lambda_1, half, lambda_1); // lambda_1 = 0.5 * (grad_xx + grad_yy + sqrt((grad_xx - grad_yy)^2 + 4 * grad_xy^2))
    cv.multiply(lambda_2, half, lambda_2); // lambda_2 = 0.5 * (grad_xx + grad_yy - sqrt((grad_xx - grad_yy)^2 + 4 * grad_xy^2))

    four_grad_xy.delete();
    sqrt.delete();
    trace.delete();

    return [lambda_1, lambda_2];
}


function get_eigenvectors(grad_xx, grad_yy, grad_xy, lambda_1, lambda_2) {

    // vec_1 = (grad_xx - lambda_1) / grad_xy
    // vec_2 = (grad_xx - lambda_2) / grad_xy

    let a = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);
    cv.subtract(grad_xx, lambda_1, a);
    cv.divide(a, grad_xy, a);
    let b = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F, new cv.Scalar(1));
    let norm = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F);
    cv.magnitude(a, b, norm);
    cv.divide(a, norm, a);
    cv.divide(b, norm, b);
    let vec_1 = new cv.MatVector();
    vec_1.push_back(a);
    vec_1.push_back(b);
    b.delete();

    cv.subtract(grad_xx, lambda_2, a);
    cv.divide(a, grad_xy, a);
    b = new cv.Mat(grad_xx.rows, grad_xx.cols, cv.CV_32F, new cv.Scalar(1));
    cv.magnitude(a, b, norm);
    cv.divide(a, norm, a);
    cv.divide(b, norm, b);
    let vec_2 = new cv.MatVector();
    vec_2.push_back(a);
    vec_2.push_back(b);

    a.delete();
    b.delete();

    return [vec_1, vec_2];
}


function get_random_position(valid_positions, threshold = 40) {

    let x = Math.floor(Math.random() * valid_positions.cols);
    let y = Math.floor(Math.random() * valid_positions.rows);
    while (valid_positions.floatPtr(y, x)[0] < threshold) {
        x = Math.floor(Math.random() * valid_positions.cols);
        y = Math.floor(Math.random() * valid_positions.rows);
    }
    // console.log(x, y, valid_positions.floatPtr(y, x)[0]);
    return [x, y];
}


function interpolate_bilinear(img, x, y) {
    let x1 = Math.floor(x);
    let x2 = Math.ceil(x);
    let y1 = Math.floor(y);
    let y2 = Math.ceil(y);
    let dx = x - x1;
    let dy = y - y1;

    if (x1 < 0) x1 = 0;
    if (x2 < 0) x2 = 0;
    if (y1 < 0) y1 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 >= img.cols) x1 = img.cols - 1;
    if (x2 >= img.cols) x2 = img.cols - 1;
    if (y1 >= img.rows) y1 = img.rows - 1;
    if (y2 >= img.rows) y2 = img.rows - 1;
    let v1 = img.floatPtr(y1, x1)[0];
    let v2 = img.floatPtr(y1, x2)[0];
    let v3 = img.floatPtr(y2, x1)[0];
    let v4 = img.floatPtr(y2, x2)[0];
    let v = (1 - dx) * (1 - dy) * v1 + dx * (1 - dy) * v2 + (1 - dx) * dy * v3 + dx * dy * v4;
    return v;
}


function RK4(f, range, y0, h = 0.01, max_iter = 1e4) {
    let line = [];
    let ts = [];
    start_t = range[0];
    end_t = range[1];
    let t = start_t;
    let message = "Failed";
    let x = y0[0];
    let y = y0[1];
    let k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y;
    for (let j = 0; j < max_iter; j++) {
        line.push([x, y]);
        ts.push(t);
        // console.log(i, j, x, y);

        [k1x, k1y] = f(t, [x, y]);

        [k2x, k2y] = f(t + h / 2, [x + h * k1x / 2, y + h * k1y / 2]);

        [k3x, k3y] = f(t + h / 2, [x + h * k2x / 2, y + h * k2y / 2]);

        [k4x, k4y] = f(t + h, [x + h * k3x, y + h * k3y]);

        x += h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6;
        y += h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6;
        t += h;

        if (t >= end_t) {
            message = "Success";
            break;
        }
    }

    return { "y": line, "t": ts, "message": message };
}


function get_tractography(vec_field, valid_positions, length = 100, num_lines = 100, grad_cutoff = 40) {

    let dx = vec_field.get(0);
    let dy = vec_field.get(1);

    let last_vec = [];

    function f(t, pos) {
        // console.log(pos);
        let d_pos_y = -interpolate_bilinear(dx, pos[0], pos[1]);
        let d_pos_x = interpolate_bilinear(dy, pos[0], pos[1]);
        if (last_vec) {
            let dot = d_pos_x * last_vec[0] + d_pos_y * last_vec[1];
            if (dot < 0) {
                d_pos_x = -d_pos_x;
                d_pos_y = -d_pos_y;
            }
        }
        last_vec = [d_pos_x, d_pos_y];
        // console.log(d_pos_x, d_pos_y);
        // d_pos_x = isNaN(d_pos_x) ? 0 : d_pos_x;
        // d_pos_y = isNaN(d_pos_y) ? 0 : d_pos_y;
        return [d_pos_x, d_pos_y];
    }

    return Array(num_lines).fill(undefined).map(async (_, i) => {
        return new Promise((resolve, reject) => {
            let line = [];
            let y0 = get_random_position(valid_positions, grad_cutoff);
            setTimeout(() => {
                try {
                    // let sol = math.solveODE(f, [0, length], y0, { "maxIter": 1e4 });
                    let sol = RK4(f, [0, length], y0, 1, 1e4);
                    console.log(sol.message);
                    line = sol.y;
                } catch (error) {
                    console.log(error);
                    line = [];
                }
                resolve(line);
            }, 0);
        });
    });
}


function clear() {
    console.log('Clearing memory');
    console.log('src_gray', src_gray);
    console.log('grad', grad);
    console.log('vec_2', vec_2);
    if (!src_gray.isDeleted()) src_gray.delete();
    if (!grad.isDeleted()) grad.delete();
    if (!vec_2.isDeleted()) {
        if (!vec_2.get(0).isDeleted()) vec_2.get(0).delete();
        if (!vec_2.get(1).isDeleted()) vec_2.get(1).delete();
        vec_2.delete();
    }
    data = null;
    offscreenCanvas.remove();
    offscreenCanvas = null;
    src_gray = null;
    grad = null;
    vec_2 = null;
}


imgElement.onload = async function () {
    try {
        // if (src_gray) {
        //     clear();
        // }

        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = imgElement.naturalWidth;
        offscreenCanvas.height = imgElement.naturalHeight;
        outputElement.width = imgElement.naturalWidth;
        outputElement.height = imgElement.naturalHeight;

        let ctx_offscreen = offscreenCanvas.getContext('2d');
        ctx_offscreen.drawImage(imgElement, 0, 0, imgElement.naturalWidth, imgElement.naturalHeight);
        let src = cv.matFromImageData(ctx_offscreen.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height));
        console.log(src);
        let ctx = outputElement.getContext('2d');
        ctx.lineWidth = parseFloat(lineWidthElement.value);
        ctx.lineCap = 'round';
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, outputElement.width, outputElement.height);
        src_gray = new cv.Mat(src.rows, src.cols, cv.CV_8UC1);
        cv.cvtColor(src, src_gray, cv.COLOR_RGBA2GRAY, 0);
        src.delete();
        let grad_xx, grad_yy, grad_xy;
        [grad_xx, grad_yy, grad_xy, grad] = get_structural_tensor(src_gray);
        let [lambda_1, lambda_2] = get_eigenvalues(grad_xx, grad_yy, grad_xy);
        let vec_1;
        [vec_1, vec_2] = get_eigenvectors(grad_xx, grad_yy, grad_xy, lambda_1, lambda_2);

        grad_xx.delete();
        grad_yy.delete();
        grad_xy.delete();
        lambda_1.delete();
        lambda_2.delete();
        vec_1.get(0).delete();
        vec_1.get(1).delete();
        vec_1.delete();


        let num_lines = parseFloat(numLinesElement.value);
        let length_line = parseFloat(lengthLineElement.value);
        if (bar instanceof ProgressBar) {
            bar.setProgress(0);
            bar.setTotal(num_lines);
            bar.restart();
        }
        else {
            bar = new ProgressBar(document.getElementById('progress-bar-container'), num_lines, clear);
        }

        data = src_gray.data;

        let lines = get_tractography(vec_2, grad, length_line, num_lines, parseFloat(gradientCutoffElement.value));
        lines.map(p => {
            p.then(line => {
                if (line.length > 0) {
                    let p1 = { x: line[0][0], y: line[0][1] };
                    for (let i = 1; i < line.length - 1; i++) {
                        let p2 = { x: line[i][0], y: line[i][1] };
                        let x = Math.floor((p1.x + p2.x) / 2);
                        let y = Math.floor((p1.y + p2.y) / 2);
                        if (x < 0) x = 0;
                        if (x >= src_gray.cols) x = src_gray.cols - 1;
                        if (y < 0) y = 0;
                        if (y >= src_gray.rows) y = src_gray.rows - 1;
                        let color = data[y * src_gray.cols + x];
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgb(${color}, ${color}, ${color})`;
                        p1 = p2;
                        ctx.stroke();
                    }
                }
                bar.setProgress(bar.progress + 1);
            });
        });
    } catch (error) {
        console.log(src_gray);
        console.log(error);
        console.error(cv.exceptionFromPtr(error).msg);
    }
};


async function loadCV() {
    console.log('OpenCV.js is ready.');
    window.cv = await cv;
    console.log(cv.getBuildInformation());
}
