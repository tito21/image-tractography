
function millis_to_time(millis) {

    let seconds = Math.floor(millis / 1000);
    let minutes = Math.floor(seconds / 60);
    let hours = Math.floor(minutes / 60);
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

class ProgressBar {
    constructor(container, total = 1, callback = () => {}) {
        this.container = container;
        this.progress = 0;
        this.total = total;
        this.callback = callback;
        this.bar = document.createElement('progress');
        this.bar.setAttribute('class', 'progress');
        this.bar.setAttribute('value', 0);
        this.bar.setAttribute('max', this.total);
        this.text = document.createElement('span');
        this.container.appendChild(this.bar);
        this.container.appendChild(this.text);
        this.start_time = Date.now();
        this.finished = false;
    }

    setProgress(progress) {
        if (this.finished) {
            return;
        }
        this.progress = progress;
        this.update();
        if (this.progress >= this.total) {
            this.bar.classList.add('done');
            this.callback();
            this.finished = true;
        }
    }

    update() {
        // console.log("Progress", this.progress);
        this.bar.setAttribute('value', this.progress);
        this.elapsed_time = Date.now() - this.start_time;
        this.estimated_time = this.elapsed_time * this.total / this.progress;
        this.text.innerHTML = `${this.progress}/${this.total} - ${millis_to_time(this.elapsed_time)} - ${millis_to_time(this.estimated_time)}`;
        // console.log("Elapsed time", this.elapsed_time);
        // console.log("Estimated time", this.estimated_time);
    }

    setTotal(total) {
        this.total = total;
        this.bar.setAttribute('max', this.total);
    }

    restart() {
        this.progress = 0;
        this.start_time = Date.now();
        this.finished = false;
        this.bar.classList.remove('done');
        this.update();
    }
}