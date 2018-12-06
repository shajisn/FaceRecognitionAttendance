$(document).ready(function() {
    let namespace = "/video";
    let video = document.querySelector("#videoElement");
    let canvas = document.querySelector("#canvasElement");
    let name = document.querySelector("#name");
    let ctx = canvas.getContext('2d');
    var localMediaStream = null;
    var timer_fn = null;

    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);

    console.log(location.protocol + '//' + document.domain + ':' + location.port + namespace);

    function sendSnapshot() {
        if (!localMediaStream) {
            return;
        }
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
        let dataURL = canvas.toDataURL('image/jpeg');
        console.log('Sending image data to server ...');
        socket.emit('input image', dataURL);

    }

    socket.on('connect', function() {
        console.log('Socket Connected!');
    });

    socket.on('response', function(data){
        console.log(data);
        name.innerHTML = data;
    });

    var constraints = {
        video: {
            width: {
                min: 640
            },
            height: {
                min: 480
            }
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        video.srcObject = stream;
        localMediaStream = stream;
        timer_fn = setInterval(function() {
            sendSnapshot();
        }, 1000);
    }).catch(function(error) {
        console.log(error);
    });
});